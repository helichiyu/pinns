import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from losses import (AugmentedLagrangian, amplitude_mask, area_ratio_residual,
                    background_loss, contour_area_ratio_loss, dynamic_contour,
                    normalized_log_amplitude_loss, soft_histogram_cdf_loss)
from model import UNet
from visualization import (plot_convergence, plot_real_space, plot_spectra,
                           plot_support, save_history, save_metrics)


def format_duration(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return "{}s".format(seconds)
    if seconds < 3600:
        return "{}m{:02d}s".format(seconds // 60, seconds % 60)
    return "{}h{:02d}m".format(seconds // 3600, (seconds % 3600) // 60)


def load_source(path, device):
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    edges = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    background = float(np.median(edges))
    source = torch.from_numpy(np.clip(background - gray, 0.0, 1.0))[None, None].to(device)
    height, width = source.shape[-2:]
    pad_height, pad_width = (-height) % 16, (-width) % 16
    top, bottom = pad_height // 2, pad_height - pad_height // 2
    left, right = pad_width // 2, pad_width - pad_width // 2
    return F.pad(source, (left, right, top, bottom)), (left, right, top, bottom)


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


def parse_args():
    parser = argparse.ArgumentParser(description="单 UNet 三损失相位恢复")
    parser.add_argument("--image", default=config.IMAGE_PATH)
    parser.add_argument("--iterations", type=int, default=config.ITERATIONS)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-every", type=int, default=config.LOG_EVERY)
    return parser.parse_args()


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    source, padding = load_source(args.image, device)
    target_amplitude = torch.abs(torch.fft.fft2(source)).detach()
    valid_amplitude = amplitude_mask(target_amplitude, config.AMPLITUDE_FLOOR)
    source_contour = dynamic_contour(source, config.CONTOUR_SIGMA, config.CONTOUR_THRESHOLD).detach()
    source_area_ratio = source_contour.mean().detach()
    model = UNet(config.BASE_CHANNELS).to(device)
    lagrangian = AugmentedLagrangian(config.SHARE_HISTOGRAM, config.SHARE_BACKGROUND,
                                     config.SHARE_AREA_RATIO, config.AL_LAMBDA_RATIO).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    current_input = random_phase_initialization(target_amplitude)
    output_dir = args.output or os.path.join(config.RESULTS_DIR, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    history = {key: [] for key in ("iteration", "total", "amplitude", "histogram", "background", "area_ratio",
                                   "r_area_ratio", "lam_area_ratio",
                                   "psnr", "ssim", "pearson_cc", "amp_cc", "phase_error", "support_iou")}

    print("设备：{}；输入：{}；轮数：{}；原图软轮廓均值：{:.2%}".format(
        device, args.image, args.iterations, source_area_ratio.item()))
    start_time = time.time()
    for iteration in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(current_input)
        amplitude = normalized_log_amplitude_loss(prediction, target_amplitude, valid_amplitude)
        histogram = soft_histogram_cdf_loss(prediction, source, config.HISTOGRAM_BINS, config.HISTOGRAM_SOFTNESS)
        contour = dynamic_contour(prediction, config.CONTOUR_SIGMA, config.CONTOUR_THRESHOLD)
        background = background_loss(prediction, contour)
        area_ratio = contour_area_ratio_loss(contour, source_area_ratio)
        area_residual = area_ratio_residual(contour, source_area_ratio)
        if iteration == 1:
            lagrangian.calibrate(amplitude, histogram, background, area_ratio)
        total = lagrangian.total(amplitude, histogram, background, area_ratio, area_residual)
        total.backward()
        optimizer.step()
        lagrangian.update_multipliers(area_residual)
        current_input = prediction.detach()

        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.iterations:
            with torch.no_grad():
                evaluation_prediction = register_to_source(prediction, source)
                evaluation_contour = dynamic_contour(evaluation_prediction, config.CONTOUR_SIGMA, config.CONTOUR_THRESHOLD)
                amp_cc, phase_error = amplitude_metrics(evaluation_prediction, source)
                values = (iteration, total.item(), amplitude.item(), histogram.item(), background.item(), area_ratio.item(),
                          area_residual.item(), lagrangian.lambda_area.item(),
                          psnr(evaluation_prediction, source), ssim(evaluation_prediction, source),
                          pearson_cc(evaluation_prediction, source), amp_cc, phase_error,
                          support_iou(evaluation_contour, source_contour))
            for key, value in zip(history, values):
                history[key].append(value)
            elapsed = time.time() - start_time
            eta = elapsed / iteration * (args.iterations - iteration)
            print("[{}/{}] total={:.3e} amp={:.3e} hist={:.3e} bg={:.3e} area={:.3e} "
                  "ssim={:.3f} iou={:.3f} elapsed={} eta={}".format(
                      iteration, args.iterations, *values[1:6],
                      values[9], values[13],
                      format_duration(elapsed), format_duration(eta)))

    with torch.no_grad():
        final_prediction = prediction.detach()
        display_prediction = register_to_source(final_prediction, source)
        final_contour = dynamic_contour(display_prediction, config.CONTOUR_SIGMA, config.CONTOUR_THRESHOLD)
    plot_real_space(source, display_prediction, padding, os.path.join(output_dir, "real_space.png"))
    plot_spectra(source, display_prediction, os.path.join(output_dir, "spectra.png"))
    plot_support(source_contour, final_contour, padding, os.path.join(output_dir, "support.png"))
    plot_convergence(history, os.path.join(output_dir, "convergence.png"))
    save_history(history, os.path.join(output_dir, "history.csv"))
    save_metrics(history, os.path.join(output_dir, "metrics.csv"))
    torch.save({"model": model.state_dict(), "history": history, "config": vars(config)},
               os.path.join(output_dir, "state.pt"))
    print("结果已保存到：{}".format(output_dir))


if __name__ == "__main__":
    main(parse_args())
