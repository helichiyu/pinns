import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config_projected_123 as config
from losses import (AugmentedLagrangian, amplitude_mask, area_ratio_residual,
                    background_loss, contour_area_ratio_loss,
                    dynamic_contour, normalized_log_amplitude_loss,
                    soft_histogram_cdf_loss)
from main import (amplitude_metrics, format_duration, pearson_cc, psnr,
                  random_phase_initialization, register_to_source, ssim,
                  support_iou)
from model_projected_123 import ProjectedUNet, full_amplitude_projection
from visualization import (plot_convergence, plot_real_space, plot_spectra,
                           plot_support, save_history, save_metrics)


def load_source_123(path, device):
    # Blue object on white background: invert the red channel for contrast.
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    red = rgb[..., 0]
    edges = np.concatenate((red[0], red[-1], red[:, 0], red[:, -1]))
    background = float(np.median(edges))
    source = torch.from_numpy(np.clip(background - red, 0.0, 1.0))[None, None].to(device)

    # Keep the existing 2x zero-padded canvas as the oversampled domain.
    height, width = source.shape[-2:]
    canvas = torch.zeros(1, 1, 2 * height, 2 * width, dtype=source.dtype, device=device)
    top, left = height // 2, width // 2
    canvas[..., top:top + height, left:left + width] = source
    pad_height, pad_width = (-canvas.shape[-2]) % 16, (-canvas.shape[-1]) % 16
    pad_top, pad_bottom = pad_height // 2, pad_height - pad_height // 2
    pad_left, pad_right = pad_width // 2, pad_width - pad_width // 2
    return F.pad(canvas, (pad_left, pad_right, pad_top, pad_bottom)), (0, 0, 0, 0)


def parse_args():
    parser = argparse.ArgumentParser(description="Projected U-Net phase retrieval for 123.png")
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
    source, padding = load_source_123(args.image, device)
    target_amplitude = torch.abs(torch.fft.fft2(source)).detach()
    valid_amplitude = amplitude_mask(target_amplitude, config.AMPLITUDE_FLOOR)
    source_contour = dynamic_contour(source, config.CONTOUR_SIGMA, config.CONTOUR_THRESHOLD).detach()
    source_area_ratio = source_contour.mean().detach()
    model = ProjectedUNet(config.BASE_CHANNELS).to(device)
    lagrangian = AugmentedLagrangian(config.SHARE_HISTOGRAM, config.SHARE_BACKGROUND,
                                     config.SHARE_AREA_RATIO, config.AL_LAMBDA_RATIO).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    current_input = random_phase_initialization(target_amplitude)
    output_dir = args.output or os.path.join(config.RESULTS_DIR, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    history = {key: [] for key in (
        "iteration", "total", "amplitude", "histogram", "background", "area_ratio",
        "r_area_ratio", "lam_area_ratio",
        "projection_amplitude", "psnr", "ssim", "pearson_cc", "amp_cc", "phase_error", "support_iou",
    )}

    print("Device: {}; input: {}; iterations: {}; canvas: {}; source contour ratio: {:.2%}".format(
        device, args.image, args.iterations, tuple(source.shape[-2:]), source_area_ratio.item()))
    start_time = time.time()
    for iteration in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        projected_input = full_amplitude_projection(current_input, target_amplitude)
        prediction = model(current_input, projected_input)
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
                evaluation_contour = dynamic_contour(
                    evaluation_prediction, config.CONTOUR_SIGMA, config.CONTOUR_THRESHOLD)
                amp_cc, phase_error = amplitude_metrics(evaluation_prediction, source)
                projection_amplitude = normalized_log_amplitude_loss(
                    projected_input, target_amplitude, valid_amplitude)
                values = (
                    iteration, total.item(), amplitude.item(), histogram.item(), background.item(), area_ratio.item(),
                    area_residual.item(), lagrangian.lambda_area.item(),
                    projection_amplitude.item(), psnr(evaluation_prediction, source), ssim(evaluation_prediction, source),
                    pearson_cc(evaluation_prediction, source), amp_cc, phase_error,
                    support_iou(evaluation_contour, source_contour),
                )
            for key, value in zip(history, values):
                history[key].append(value)
            elapsed = time.time() - start_time
            eta = elapsed / iteration * (args.iterations - iteration)
            print("[{}/{}] total={:.3e} amp={:.3e} hist={:.3e} bg={:.3e} area={:.3e} "
                  "ssim={:.3f} iou={:.3f} elapsed={} eta={}".format(
                      iteration, args.iterations, values[1], values[2], values[3], values[4], values[5],
                      values[10], values[14], format_duration(elapsed), format_duration(eta)))

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
    torch.save({
        "model": model.state_dict(),
        "history": history,
        "config": vars(config),
        "architecture": "two-input ProjectedUNet with complete-amplitude projection",
    }, os.path.join(output_dir, "state.pt"))
    print("Results saved to: {}".format(output_dir))


if __name__ == "__main__":
    main(parse_args())
