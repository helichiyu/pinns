import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_blur(image, sigma):
    radius = int(math.ceil(3 * sigma))
    axis = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-(axis[:, None].square() + axis[None, :].square()) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)
    return F.conv2d(image, kernel, padding=radius)


def amplitude_mask(target_amplitude, relative_floor):
    mask = target_amplitude >= target_amplitude.amax() * relative_floor
    mask = mask.clone()
    mask[..., 0, 0] = False
    return mask


def normalized_log_amplitude_loss(prediction, target_amplitude, valid_mask):
    prediction_amplitude = torch.abs(torch.fft.fft2(prediction))
    scale = target_amplitude.amax().detach().clamp_min(1e-8)
    prediction_log = torch.log1p(prediction_amplitude / scale)
    target_log = torch.log1p(target_amplitude / scale)
    return F.mse_loss(prediction_log[valid_mask], target_log[valid_mask])


def soft_histogram_cdf_loss(prediction, target, bins, softness):
    centers = torch.linspace(0, 1, bins, device=prediction.device, dtype=prediction.dtype)
    prediction_values = prediction.flatten()
    target_values = target.flatten()
    prediction_cdf = torch.sigmoid((centers[:, None] - prediction_values[None, :]) / softness).mean(dim=1)
    target_cdf = torch.sigmoid((centers[:, None] - target_values[None, :]) / softness).mean(dim=1)
    return F.mse_loss(prediction_cdf, target_cdf)


def dynamic_contour(prediction, sigma, threshold):
    blurred = gaussian_blur(prediction, sigma)
    baseline = blurred.amin(dim=(-2, -1), keepdim=True)
    contrast = blurred - baseline
    peak = contrast.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return torch.sigmoid(12.0 * (contrast / peak - threshold))


def background_loss(prediction, contour):
    outside = 1.0 - contour.detach()
    return torch.mean((outside * prediction).square())


def contour_area_ratio_loss(contour, target_ratio):
    return F.mse_loss(contour.mean(), target_ratio)


def area_ratio_residual(contour, target_ratio):
    """占比约束的带符号残差 contour.mean() - target_ratio。

    偏小为负、偏大为正，故单个乘子即可双侧约束（不需要容差与 relu）：
    占比偏小 ⇒ r<0 ⇒ λ 变负 ⇒ 最小化 λ·r 即把 r 推大，撑开轮廓；偏大则反向。
    contour_area_ratio_loss 恰为该残差的平方，即对应的罚项。
    """
    return contour.mean() - target_ratio


class AugmentedLagrangian(nn.Module):
    """AL-PINNs 增广拉格朗日（Son et al., arXiv:2205.01059），只用于轮廓占比约束。

    论文的目标函数（式 5）：

        L_λ(θ) = ‖N u - f‖         # 目标
               + β ‖T u - g‖²      # 罚项：β 是预先给定的固定常数
               + ⟨λ, T u - g⟩      # 乘子项：λ 由梯度上升每步更新

    映射到本项目：

        L = L_amp                                # 目标，权重恒 1
          + β_hist · L_hist                      # 统计先验
          + β_bg   · L_bg                        # 背景：只有罚项，见下
          + β_area · L_area + λ_area · r_area    # 占比：完整增广拉格朗日

    背景为何只有罚项、没有乘子：论文的边界残差 T u - g 是带符号的、能穿过零，
    λ 才会自己停下来（Lemma 3.5 的有界性）。而背景残差 mean((1-contour)·pred)
    恒为正——pred 经 Sigmoid 恒正、outside 恒非负，永远到不了零。配上乘子后
    λ_bg 单调无界增长，把「整图全黑」推成最优解，实测导致 support_iou 从 0.8 崩到 0.1。
    占比残差 r_area = contour.mean() - target 能穿过零，故只有它适合配乘子。

    各项权重 β 由首次前向按目标贡献比自动标定，见 calibrate。
    λ 与 β 都是不参与反向传播的 buffer。
    """

    def __init__(self, share_histogram, share_background, share_area_ratio, lambda_ratio):
        super().__init__()
        self.shares = (share_histogram, share_background, share_area_ratio)
        self.lambda_ratio = lambda_ratio
        self.register_buffer("betas", torch.zeros(3))   # 顺序：直方图、背景、占比
        self.register_buffer("lambda_area", torch.zeros(()))
        self.register_buffer("lambda_lr", torch.zeros(()))

    def total(self, amplitude, histogram, background, area_ratio, area_residual):
        weighted = self.betas * torch.stack([histogram, background, area_ratio])
        return amplitude + weighted.sum() + self.lambda_area * area_residual

    @torch.no_grad()
    def calibrate(self, amplitude, histogram, background, area_ratio):
        """首步自动标定：β_i = SHARE_i × L_amp,0 / L_i,0。

        振幅是唯一的硬数据，权重恒为 1；其余三项按各自的可信度取一个目标贡献比
        （见 config 的 SHARE_* 注释）。换图片时各项损失值都变，算出的 β 跟着变，
        但贡献占比不变，故 SHARE_* 是全局常数而非每图参数。

        注意不能让三项与振幅等权：背景损失天生就小，等权会把 β_bg 抬高约 140 倍，
        使「压黑整图」的力压过振幅目标而塌缩。振幅必须主导。
        """
        anchor = amplitude.detach().clamp_min(1e-12)
        initial = torch.stack([loss.detach() for loss in (histogram, background, area_ratio)])
        shares = torch.tensor(self.shares, device=initial.device, dtype=initial.dtype)
        self.betas.copy_(shares * anchor / initial.clamp_min(1e-12))
        self.lambda_lr.copy_(self.betas[2] * self.lambda_ratio)
        self.lambda_area.zero_()

    @torch.no_grad()
    def update_multipliers(self, area_residual):
        """梯度上升：λ ← λ + η_λ · ∂L/∂λ，而 ∂L/∂λ = r_area。

        λ 不限号：占比偏小时 r_area < 0 使 λ 变负，最小化 λ·r 即把占比推大
        （撑开轮廓）；偏大时反向。故单个乘子天然双侧约束，无需容差。
        r_area → 0 时 λ 自动停止变化，不会像恒正残差那样无界累积。
        """
        self.lambda_area.add_(self.lambda_lr * area_residual.detach())
