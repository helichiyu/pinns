import argparse
import json
import os
import random
import sys
import threading
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from losses import (UncertaintyWeights, amplitude_mask, background_loss,
                    masked_histogram_quantile_loss, normalized_log_amplitude_loss,
                    quantile_ranks, source_contour, topk_contour)
from model import UNet
from visualization import (plot_convergence, plot_real_space, plot_spectra,
                           plot_support, save_history, save_metrics)

# Prefix for structured metric lines parsed by the web server. Hidden from the
# human-readable terminal panel on the frontend side.
METRIC_PREFIX = "__METRIC__"
RESULT_PREFIX = "__RESULT__"

# 末三列是 Kendall 可学习权重的对数方差 s_i，只入 CSV，不进收敛图。
HISTORY_KEYS = ("iteration", "total", "amplitude", "histogram", "background",
                "psnr", "ssim", "pearson_cc", "amp_cc", "support_iou",
                "s_amplitude", "s_histogram", "s_background")


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
        dimension (FFT oversampling condition). 可以是小数（如 1.5）；扩大后的
        边长向下取整，随后还要 pad 到 16 的倍数，所以实际倍率会略高于名义值。
    """
    channel = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0

    edges = np.concatenate((channel[0], channel[-1], channel[:, 0], channel[:, -1]))
    background = float(np.median(edges))
    source = torch.from_numpy(np.clip(background - channel, 0.0, 1.0))[None, None].to(device)

    height, width = source.shape[-2:]
    if expand > 1:
        new_h, new_w = int(expand * height), int(expand * width)
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


def patterson_initialization(target_amplitude, valid_amplitude):
    """Patterson map 作为首轮输入：P = IFFT(|F|² · mask)，即振幅取平方、相位全置零。

    P 数学上就是物体自身的自相关函数，不需要相位就能从实测振幅直接算出，比随机
    相位携带真实结构信息（物体尺寸、内部间距）。

    mask 复用振幅损失的低频圆盘，语义一致：输入里看不到的低频，损失里也不罚。
    真正的自相关恒非负，但挖掉频谱分量后 IFFT 不再保证这一点（实测约一半像素
    为负），所以用 min-max 归一化而不是截零。

    fftshift 把原点从 [0, 0] 搬到画布中心：不 shift 的话周期延拓会把自相关切成
    四块贴在四个角，而源图物体居中、第 2 轮起的输入（上一轮输出）也居中。

    注意 expand=1 时自相关会卷绕自叠——自相关支撑是物体的 2 倍，画布不够大就绕
    回来叠自己，要干净的 Patterson 需要 expand≥2。
    """
    intensity = target_amplitude.square() * valid_amplitude
    patterson = torch.fft.fftshift(torch.real(torch.fft.ifft2(intensity)), dim=(-2, -1))
    low, high = patterson.amin(), patterson.amax()
    return (patterson - low) / (high - low).clamp_min(1e-12)


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


def amplitude_cc_metric(prediction, source):
    target_spectrum = torch.fft.fft2(source)
    prediction_spectrum = torch.fft.fft2(prediction)
    target_amplitude = torch.abs(target_spectrum)
    prediction_amplitude = torch.abs(prediction_spectrum)
    phase_difference = torch.angle(target_spectrum) - torch.angle(prediction_spectrum)
    amplitude_cc = (target_amplitude * prediction_amplitude * torch.cos(phase_difference)).sum()
    return amplitude_cc.div(target_amplitude.norm() * prediction_amplitude.norm() + 1e-12).item()


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


# 前端「结束当前」会往 stdin 写一行 stop：训练在下一轮开头跳出循环，
# 用已有 history 正常出图，不是 kill 进程。
_stop_requested = threading.Event()


def watch_stop_signal():
    """后台线程读 stdin，收到 stop 就置位停止标志。"""
    def loop():
        for line in sys.stdin:
            if line.strip() == "stop":
                _stop_requested.set()
                return
    threading.Thread(target=loop, daemon=True).start()


def parse_args():
    parser = argparse.ArgumentParser(description="Unified single-UNet phase retrieval")
    parser.add_argument("--image", default=config.IMAGE_PATH)
    parser.add_argument("--expand", type=float, default=1,
                        help="canvas expansion factor (1 = no expansion, 可为小数如 1.5)")
    parser.add_argument("--contour-sigma", type=float, default=config.CONTOUR_SIGMA,
                        help="Gaussian blur radius for contour extraction")
    parser.add_argument("--contour-threshold", type=float, default=config.CONTOUR_THRESHOLD,
                        help="源图轮廓的相对峰值阈值（只用于标定源图 mask 与像素数 k）")
    parser.add_argument("--share-histogram", type=float, default=config.SHARE_HISTOGRAM,
                        help="轮廓内直方图项初始贡献占振幅项的比例")
    parser.add_argument("--share-background", type=float, default=config.SHARE_BACKGROUND,
                        help="背景项初始贡献占振幅项的比例")
    parser.add_argument("--iterations", type=int, default=config.ITERATIONS)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-every", type=int, default=config.LOG_EVERY)
    parser.add_argument("--stream-metrics", action="store_true",
                        help="emit __METRIC__ JSON lines for the web server")
    return parser.parse_args()


def main(args):
    if args.stream_metrics:
        watch_stop_signal()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    source, padding = load_source(args.image, device, args.expand)
    contour_sigma = args.contour_sigma
    contour_threshold = args.contour_threshold
    target_amplitude = torch.abs(torch.fft.fft2(source)).detach()
    valid_amplitude = amplitude_mask(target_amplitude, config.AMPLITUDE_DC_RADIUS)
    source_mask = source_contour(source, contour_sigma, contour_threshold)
    contour_pixels = int(source_mask.sum().item())
    if contour_pixels == 0:
        raise SystemExit("源图轮廓为空（阈值 {} 过高），无法标定像素数 k。".format(contour_threshold))
    source_area_ratio = contour_pixels / source_mask.numel()
    if config.HISTOGRAM_BINS > contour_pixels:
        raise SystemExit("直方图箱数 {} 超过轮廓像素数 {}，分位点会重复。".format(
            config.HISTOGRAM_BINS, contour_pixels))
    ranks = quantile_ranks(contour_pixels, config.HISTOGRAM_BINS, device)
    target_quantiles = source[source_mask > 0.5].sort().values[ranks].detach()
    model = UNet(config.BASE_CHANNELS).to(device)
    weights = UncertaintyWeights(args.share_histogram, args.share_background).to(device)
    # s_i 与网络参数进同一个 Adam、同一个学习率（Kendall 原文做法）。
    optimizer = torch.optim.Adam(list(model.parameters()) + list(weights.parameters()),
                                 lr=config.LEARNING_RATE)
    current_input = patterson_initialization(target_amplitude, valid_amplitude)
    # expand 是浮点，整数值去掉尾随 .0，让目录名保持 x1 / x2 而不是 x1.0 / x2.0
    expand_tag = "{:g}".format(args.expand)
    tag = "{}_x{}".format(os.path.splitext(os.path.basename(args.image))[0], expand_tag)
    # 目录名此刻定下（时间戳＝开始时间），但等训练结束真要写文件时才创建，
    # 避免失败或中途终止在 results/ 里留下空目录。
    output_dir = args.output or os.path.join(config.RESULTS_DIR,
                                             datetime.now().strftime("run_{}_%Y%m%d_%H%M%S".format(tag)))
    history = {key: [] for key in HISTORY_KEYS}

    print("Device: {}; image: {}; expand: {}x; canvas: {}; iterations: {}".format(
        device, args.image, expand_tag, tuple(source.shape[-2:]), args.iterations))
    print("Source contour ratio: {:.2%} ({} px)".format(source_area_ratio, contour_pixels))
    start_time = time.time()
    prediction = None
    stopped_early = False
    for iteration in range(1, args.iterations + 1):
        # history 非空才允许提前结束，保证出图至少有一个数据点。
        if _stop_requested.is_set() and history["iteration"]:
            stopped_early = True
            print("收到结束指令，在第 {} 轮停止，按已有结果出图。".format(iteration - 1))
            break
        optimizer.zero_grad(set_to_none=True)
        prediction = model(current_input)
        amplitude = normalized_log_amplitude_loss(prediction, target_amplitude, valid_amplitude)
        contour = topk_contour(prediction, contour_sigma, contour_pixels)
        histogram = masked_histogram_quantile_loss(prediction, contour, target_quantiles, ranks)
        background = background_loss(prediction, contour)
        if iteration == 1:
            weights.initialize(amplitude, histogram, background)
        total = weights.total(amplitude, histogram, background)
        total.backward()
        optimizer.step()
        current_input = prediction.detach()

        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.iterations:
            with torch.no_grad():
                evaluation_prediction = register_to_source(prediction, source)
                evaluation_contour = topk_contour(evaluation_prediction, contour_sigma, contour_pixels)
                amp_cc = amplitude_cc_metric(evaluation_prediction, source)
                # 本块在 optimizer.step() 之后，所以 s_i 比 total 领先一个 Adam 步
                # （差约 1e-4，不影响分析）。每轮同步 s 到 CPU 会拖慢训练，不值得。
                log_variance = weights.log_variance.tolist()
                # 前端实时曲线用归一化损失 L̂_i = L_i / L_i,0：三条线同起点 1.0，往下即在降。
                # 不能用 total（含 +s_i、随 s 线性滑向 -∞）也不能用加权和
                # （不动点被钉在 Σw_i 附近，测的是 s 的错配而非损失水平）。
                normalized = (torch.stack([amplitude, histogram, background])
                              / weights.initial).tolist()
                values = (iteration, total.item(), amplitude.item(), histogram.item(), background.item(),
                          psnr(evaluation_prediction, source), ssim(evaluation_prediction, source),
                          pearson_cc(evaluation_prediction, source), amp_cc,
                          support_iou(evaluation_contour, source_mask),
                          *log_variance)
            for key, value in zip(history, values):
                history[key].append(value)
            elapsed = time.time() - start_time
            eta = elapsed / iteration * (args.iterations - iteration)
            ssim_value = values[6]
            iou = values[9]
            print("[{}/{}] total={:.3e} amp={:.3e} hist={:.3e} bg={:.3e} "
                  "ssim={:.3f} iou={:.3f} elapsed={} eta={}".format(
                      iteration, args.iterations, *values[1:5],
                      ssim_value, iou,
                      format_duration(elapsed), format_duration(eta)))
            emit_metric(args.stream_metrics, iteration=iteration, losses=normalized,
                        iou=iou, ssim=ssim_value)

    with torch.no_grad():
        final_prediction = prediction.detach()
        display_prediction = register_to_source(final_prediction, source)
        final_contour = topk_contour(display_prediction, contour_sigma, contour_pixels)
    os.makedirs(output_dir, exist_ok=True)
    plot_real_space(source, display_prediction, padding, os.path.join(output_dir, "real_space.png"))
    plot_spectra(source, display_prediction, os.path.join(output_dir, "spectra.png"))
    plot_support(source_mask, final_contour, padding, os.path.join(output_dir, "support.png"))
    plot_convergence(history, os.path.join(output_dir, "convergence.png"))
    save_history(history, os.path.join(output_dir, "history.csv"))
    save_metrics(history, os.path.join(output_dir, "metrics.csv"))
    torch.save({"model": model.state_dict(), "history": history, "config": vars(config)},
               os.path.join(output_dir, "state.pt"))
    print("Results saved to: {}".format(output_dir))
    if args.stream_metrics:
        print(RESULT_PREFIX + json.dumps({"output_dir": output_dir.replace("\\", "/"),
                                          "stopped_early": stopped_early}), flush=True)


if __name__ == "__main__":
    main(parse_args())
