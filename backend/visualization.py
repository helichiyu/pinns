import csv

import matplotlib

# 服务端在工作线程里出图，交互式后端（qtagg）在非主线程会失败，必须先切 Agg。
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.formatter.use_mathtext"] = False


def tensor_to_numpy(image, padding):
    left, right, top, bottom = padding
    height, width = image.shape[-2:]
    image = image[..., top:height - bottom if bottom else height, left:width - right if right else width]
    return image.squeeze().detach().cpu().numpy()


def plot_real_space(source, prediction, padding, save_path):
    source = tensor_to_numpy(source, padding)
    prediction = tensor_to_numpy(prediction, padding)
    error = np.abs(prediction - source)
    overlay = np.zeros((*source.shape, 3), dtype=np.float32)
    overlay[..., 0] = source
    overlay[..., 1] = prediction
    overlay[..., 2] = prediction
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panels = ((source, "原图（反转后，背景为 0）", "gray"),
              (prediction, "最终输出（背景为 0）", "gray"),
              (overlay, "叠加图（原图红，输出青）", None),
              (error, "绝对误差", "hot"))
    for axis, (image, title, cmap) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap == "gray" else None)
        axis.set_title(title, fontsize=13)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spectra(source, prediction, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, image, title in zip(axes, (source, prediction), ("原图振幅谱", "最终输出振幅谱")):
        amplitude = torch.fft.fftshift(torch.abs(torch.fft.fft2(image)))
        axis.imshow(np.log1p(amplitude.squeeze().detach().cpu().numpy()), cmap="magma")
        axis.set_title(title, fontsize=13)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_support(source_contour, prediction_contour, padding, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    panels = ((source_contour, "原图硬轮廓（仅展示/像素数基准）"),
              (prediction_contour, "最终输出的 top-k 硬轮廓"))
    for axis, (contour, title) in zip(axes, panels):
        axis.imshow(tensor_to_numpy(contour, padding), cmap="gray", vmin=0, vmax=1)
        axis.set_title(title, fontsize=13)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_source_contour_explanation(source, contour, padding, ratio, save_path):
    source_image = tensor_to_numpy(source, padding)
    contour_image = tensor_to_numpy(contour, padding)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    panels = ((source_image, "处理后的原图（灰度、反转、扩画布、pad）"),
              (contour_image, "二值化轮廓 mask（占比 {:.2%}）".format(ratio)))
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title, fontsize=13)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(history, save_path):
    panels = (("total", "总损失"), ("amplitude", "振幅损失"), ("histogram", "轮廓内直方图损失"),
              ("background", "背景损失"),
              ("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("pearson_cc", "Pearson CC"),
              ("amp_cc", "振幅域 CC"), ("phase_error", "平均相位误差 (rad)"), ("support_iou", "粗轮廓 IoU"))
    ncols = 4
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 4))
    log_keys = ("amplitude", "histogram", "background")
    # total 含振幅项，量级约 1e-4，需要科学计数法的线性轴。
    small_linear_keys = ("total",)
    for axis, (key, title) in zip(axes.flat, panels):
        axis.plot(history["iteration"], history[key], color="#2E86AB", lw=2)
        axis.set_xlabel("迭代轮数")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        if key in log_keys:
            axis.set_yscale("log")
            axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: "{:.0e}".format(value)))
            axis.yaxis.set_minor_formatter(NullFormatter())
        elif key in small_linear_keys:
            # 残差与乘子可为负、量级又极小，用科学计数法的线性轴。
            axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: "{:.1e}".format(value)))
        else:
            axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: "{:.3f}".format(value)))
    for axis in axes.flat[len(panels):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_history(history, save_path):
    with open(save_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))


def save_metrics(history, save_path):
    keys = ("psnr", "ssim", "pearson_cc", "amp_cc", "phase_error", "support_iou")
    with open(save_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(("指标", "末轮值"))
        for key in keys:
            writer.writerow((key, history[key][-1]))
