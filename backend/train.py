import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from losses import (CalibratedWeights, amplitude_mask, background_loss,
                    contour_area_ratio_loss, dynamic_contour,
                    normalized_log_amplitude_loss, soft_histogram_cdf_loss)
from model import UNet
from visualization import (plot_convergence, plot_real_space, plot_spectra,
                           plot_support, save_history, save_metrics)

# Prefix for structured metric lines parsed by the web server. Hidden from the
# human-readable terminal panel on the frontend side.
METRIC_PREFIX = "__METRIC__"
RESULT_PREFIX = "__RESULT__"

HISTORY_KEYS = ("iteration", "total", "amplitude", "histogram", "background", "area_ratio",
                "psnr", "ssim", "pearson_cc", "amp_cc", "phase_error", "support_iou")


def format_duration(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return "{}s".format(seconds)
    if seconds < 3600:
        return "{}m{:02d}s".format(seconds // 60, seconds % 60)
    return "{}h{:02d}m".format(seconds // 3600, (seconds % 3600) // 60)


def load_source(path, device, expand=1):
    """Load an image as an inverted density tensor on a padded canvas.

    expand:
        Canvas expansion factor (1 = no expansion). When > 1 the object is
        centered on a zero-filled canvas that is `expand` times larger per
        dimension (FFT oversampling condition).
    """
    channel = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0

    edges = np.concatenate((channel[0], channel[-1], channel[:, 0], channel[:, -1]))
    background = float(np.median(edges))
    source = torch.from_numpy(np.clip(background - channel, 0.0, 1.0))[None, None].to(device)

    height, width = source.shape[-2:]
    if expand > 1:
        new_h, new_w = expand * height, expand * width
        canvas = torch.zeros(1, 1, new_h, new_w, dtype=source.dtype, device=device)
        top, left = (new_h - height) // 2, (new_w - width) // 2
        canvas[..., top:top + height, left:left + width] = source
        source = canvas

    height, width = source.shape[-2:]
    pad_h, pad_w = (-height) % 16, (-width) % 16
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    source = F.pad(source, (left, right, top, bottom))

    if expand > 1:
        return source, (0, 0, 0, 0)
    return source, (left, right, top, bottom)


def random_phase_initialization(target_amplitude):
    phase = torch.rand_like(target_amplitude) * (2 * np.pi) - np.pi
    initial = torch.real(torch.fft.ifft2(target_amplitude * torch.exp(1j * phase))).abs()
    return initial / initial.amax().clamp_min(1e-6)


def psnr(prediction, source):
    return (-10.0 * torch.log10(F.mse_loss(prediction, source) + 1e-12)).item()


def ssim(prediction, source):
    mean_prediction = F.avg_pool2d(prediction, 11, stride=1, padding=5)
    mean_source = F.avg_pool2d(source, 11, stride=1, padding=5)
    variance_prediction = F.avg_pool2d(prediction.square(), 11, 1, 5) - mean_prediction.square()
    variance_source = F.avg_pool2d(source.square(), 11, 1, 5) - mean_source.square()
    covariance = F.avg_pool2d(prediction * source, 11, 1, 5) - mean_prediction * mean_source
    return (((2 * mean_prediction * mean_source + 0.01**2) * (2 * covariance + 0.03**2)) /
            ((mean_prediction.square() + mean_source.square() + 0.01**2) *
             (variance_prediction + variance_source + 0.03**2) + 1e-12)).mean().item()


def pearson_cc(prediction, source):
    prediction = prediction.flatten() - prediction.mean()
    source = source.flatten() - source.mean()
    return (prediction * source).sum().div(prediction.norm() * source.norm() + 1e-12).item()


def amplitude_metrics(prediction, source):
    target_spectrum = torch.fft.fft2(source)
    prediction_spectrum = torch.fft.fft2(prediction)
    target_amplitude = torch.abs(target_spectrum)
    prediction_amplitude = torch.abs(prediction_spectrum)
    phase_difference = torch.angle(target_spectrum) - torch.angle(prediction_spectrum)
    amplitude_cc = (target_amplitude * prediction_amplitude * torch.cos(phase_difference)).sum()
    amplitude_cc = amplitude_cc.div(target_amplitude.norm() * prediction_amplitude.norm() + 1e-12).item()
    phase_error = torch.arccos(torch.cos(phase_difference)).mean().item()
    return amplitude_cc, phase_error


def support_iou(prediction_contour, source_contour):
    prediction_mask = prediction_contour > 0.5
    source_mask = source_contour > 0.5
    return (prediction_mask & source_mask).sum().div((prediction_mask | source_mask).sum() + 1e-12).item()


def register_to_source(prediction, source):
    source_spectrum = torch.fft.fft2(source)
    _, _, height, width = prediction.shape
    best_score = -float("inf")
    best_prediction = prediction
    for flipped in (False, True):
        candidate = torch.flip(prediction, dims=(-2, -1)) if flipped else prediction
        correlation = torch.fft.ifft2(source_spectrum.conj() * torch.fft.fft2(candidate)).real
        shift_y, shift_x = divmod(correlation.argmax().item(), width)
        if shift_y > height // 2:
            shift_y -= height
        if shift_x > width // 2:
            shift_x -= width
        for shift in ((shift_y, shift_x), (-shift_y, -shift_x)):
            aligned = torch.roll(candidate, shifts=shift, dims=(-2, -1))
            score = (aligned * source).sum().item()
            if score > best_score:
                best_score = score
                best_prediction = aligned
    return best_prediction


def emit_metric(stream, **kwargs):
    if stream:
        print(METRIC_PREFIX + json.dumps(kwargs), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified single-UNet phase retrieval")
    parser.add_argument("--image", default=config.IMAGE_PATH)
    parser.add_argument("--expand", type=int, default=1, help="canvas expansion factor (1 = no expansion)")
    parser.add_argument("--contour-sigma", type=float, default=config.CONTOUR_SIGMA,
                        help="Gaussian blur radius for contour extraction")
    parser.add_argument("--contour-threshold", type=float, default=config.CONTOUR_THRESHOLD,
                        help="relative peak threshold for contour extraction")
    parser.add_argument("--iterations", type=int, default=config.ITERATIONS)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-every", type=int, default=config.LOG_EVERY)
    parser.add_argument("--stream-metrics", action="store_true",
                        help="emit __METRIC__ JSON lines for the web server")
    return parser.parse_args()


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    source, padding = load_source(args.image, device, args.expand)
    contour_sigma = args.contour_sigma
    contour_threshold = args.contour_threshold
    target_amplitude = torch.abs(torch.fft.fft2(source)).detach()
    valid_amplitude = amplitude_mask(target_amplitude, config.AMPLITUDE_FLOOR)
    source_contour = dynamic_contour(source, contour_sigma, contour_threshold).detach()
    source_area_ratio = source_contour.mean().detach()
    model = UNet(config.BASE_CHANNELS).to(device)
    weights = CalibratedWeights(config.SHARE_HISTOGRAM, config.SHARE_BACKGROUND,
                                config.SHARE_AREA_RATIO).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    current_input = random_phase_initialization(target_amplitude)
    tag = "{}_x{}".format(os.path.splitext(os.path.basename(args.image))[0], args.expand)
    output_dir = args.output or os.path.join(config.RESULTS_DIR,
                                             datetime.now().strftime("run_{}_%Y%m%d_%H%M%S".format(tag)))
    os.makedirs(output_dir, exist_ok=True)
    history = {key: [] for key in HISTORY_KEYS}

    print("Device: {}; image: {}; expand: {}x; canvas: {}; iterations: {}".format(
        device, args.image, args.expand, tuple(source.shape[-2:]), args.iterations))
    print("Source contour ratio: {:.2%}".format(source_area_ratio.item()))
    start_time = time.time()
    prediction = None
    for iteration in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(current_input)
        amplitude = normalized_log_amplitude_loss(prediction, target_amplitude, valid_amplitude)
        histogram = soft_histogram_cdf_loss(prediction, source, config.HISTOGRAM_BINS, config.HISTOGRAM_SOFTNESS)
        contour = dynamic_contour(prediction, contour_sigma, contour_threshold)
        background = background_loss(prediction, contour)
        area_ratio = contour_area_ratio_loss(contour, source_area_ratio)
        if iteration == 1:
            weights.calibrate(amplitude, histogram, background, area_ratio)
        total = weights.total(amplitude, histogram, background, area_ratio)
        total.backward()
        optimizer.step()
        current_input = prediction.detach()

        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.iterations:
            with torch.no_grad():
                evaluation_prediction = register_to_source(prediction, source)
                evaluation_contour = dynamic_contour(evaluation_prediction, contour_sigma, contour_threshold)
                amp_cc, phase_error = amplitude_metrics(evaluation_prediction, source)
                values = (iteration, total.item(), amplitude.item(), histogram.item(), background.item(), area_ratio.item(),
                          psnr(evaluation_prediction, source), ssim(evaluation_prediction, source),
                          pearson_cc(evaluation_prediction, source), amp_cc, phase_error,
                          support_iou(evaluation_contour, source_contour))
            for key, value in zip(history, values):
                history[key].append(value)
            elapsed = time.time() - start_time
            eta = elapsed / iteration * (args.iterations - iteration)
            iou = values[11]
            print("[{}/{}] total={:.3e} amp={:.3e} hist={:.3e} bg={:.3e} area={:.3e} "
                  "ssim={:.3f} iou={:.3f} elapsed={} eta={}".format(
                      iteration, args.iterations, *values[1:6],
                      values[7], iou,
                      format_duration(elapsed), format_duration(eta)))
            emit_metric(args.stream_metrics, iteration=iteration, total=total.item(),
                        iou=iou, ssim=values[7])

    with torch.no_grad():
        final_prediction = prediction.detach()
        display_prediction = register_to_source(final_prediction, source)
        final_contour = dynamic_contour(display_prediction, contour_sigma, contour_threshold)
    plot_real_space(source, display_prediction, padding, os.path.join(output_dir, "real_space.png"))
    plot_spectra(source, display_prediction, os.path.join(output_dir, "spectra.png"))
    plot_support(source_contour, final_contour, padding, os.path.join(output_dir, "support.png"))
    plot_convergence(history, os.path.join(output_dir, "convergence.png"))
    save_history(history, os.path.join(output_dir, "history.csv"))
    save_metrics(history, os.path.join(output_dir, "metrics.csv"))
    torch.save({"model": model.state_dict(), "history": history, "config": vars(config)},
               os.path.join(output_dir, "state.pt"))
    print("Results saved to: {}".format(output_dir))
    if args.stream_metrics:
        print(RESULT_PREFIX + json.dumps({"output_dir": output_dir.replace("\\", "/")}), flush=True)


if __name__ == "__main__":
    main(parse_args())
