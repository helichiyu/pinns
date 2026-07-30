import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from losses import (amplitude_mask, background_loss, dynamic_contour,
                    normalized_log_amplitude_loss, soft_histogram_cdf_loss)
from model import UNet
from visualization import (plot_convergence, plot_real_space, plot_spectra,
                           plot_support, save_history)


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
    model = UNet(config.BASE_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    current_input = random_phase_initialization(target_amplitude)
    output_dir = args.output or os.path.join(config.RESULTS_DIR, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    history = {key: [] for key in ("iteration", "total", "amplitude", "histogram", "background", "psnr", "ssim")}

    print("设备：{}；输入：{}；轮数：{}".format(device, args.image, args.iterations))
    for iteration in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(current_input)
        amplitude = normalized_log_amplitude_loss(prediction, target_amplitude, valid_amplitude)
        histogram = soft_histogram_cdf_loss(prediction, source, config.HISTOGRAM_BINS, config.HISTOGRAM_SOFTNESS)
        contour = dynamic_contour(prediction, config.CONTOUR_SIGMA, config.CONTOUR_THRESHOLD)
        background = background_loss(prediction, contour)
        total = (config.WEIGHT_AMPLITUDE * amplitude + config.WEIGHT_HISTOGRAM * histogram +
                 config.WEIGHT_BACKGROUND * background)
        total.backward()
        optimizer.step()
        current_input = prediction.detach()

        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.iterations:
            with torch.no_grad():
                values = (iteration, total.item(), amplitude.item(), histogram.item(), background.item(),
                          psnr(prediction, source), ssim(prediction, source))
            for key, value in zip(history, values):
                history[key].append(value)
            print("[{}/{}] total={:.3e} amp={:.3e} hist={:.3e} bg={:.3e} psnr={:.2f} ssim={:.3f}".format(
                iteration, args.iterations, *values[1:]))

    with torch.no_grad():
        final_prediction = prediction.detach()
        final_contour = dynamic_contour(final_prediction, config.CONTOUR_SIGMA, config.CONTOUR_THRESHOLD)
    plot_real_space(source, final_prediction, padding, os.path.join(output_dir, "real_space.png"))
    plot_spectra(source, final_prediction, os.path.join(output_dir, "spectra.png"))
    plot_support(final_contour, padding, os.path.join(output_dir, "support.png"))
    plot_convergence(history, os.path.join(output_dir, "convergence.png"))
    save_history(history, os.path.join(output_dir, "history.csv"))
    torch.save({"model": model.state_dict(), "history": history, "config": vars(config)},
               os.path.join(output_dir, "state.pt"))
    print("结果已保存到：{}".format(output_dir))


if __name__ == "__main__":
    main(parse_args())
