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


def background_residual(prediction, contour):
    """背景约束的一阶残差 mean((1-contour)·pred)，供增广拉格朗日的乘子项使用。

    与 background_loss 是同一个约束（轮廓外应为 0）的两种度量：
    background_loss = mean(逐像素残差²) 对应论文的罚项 β‖C‖²，
    本函数 = mean(逐像素残差) 对应论文的乘子项 ⟨λ, C⟩。
    因 pred 与 outside 均非负，该残差恒 ≥ 0，且为 0 当且仅当约束满足。
    """
    outside = 1.0 - contour.detach()
    return (outside * prediction).mean()


def area_ratio_residual(contour, target_ratio):
    """占比约束的带符号残差 contour.mean() - target_ratio。

    偏小为负、偏大为正，故单个乘子即可双侧约束（不需要容差与 relu）：
    占比偏小 ⇒ r<0 ⇒ λ 变负 ⇒ 最小化 λ·r 即把 r 推大，撑开轮廓；偏大则反向。
    contour_area_ratio_loss 恰为该残差的平方，即对应的罚项。
    """
    return contour.mean() - target_ratio


class AugmentedLagrangian(nn.Module):
    """AL-PINNs 增广拉格朗日（Son et al., arXiv:2205.01059）。

    论文的目标函数（式 5，本文件按其形式实现）：

        L_λ(θ) = ‖N u - f‖         # 目标：振幅损失
               + β ‖T u - g‖²      # 罚项：固定罚参数 β，论文强调 β 必须是预先给定的常数
               + ⟨λ, T u - g⟩      # 乘子项：λ 由梯度上升每步更新

    映射到本项目：目标为振幅损失（权重恒 1），背景与占比为两个约束，直方图为固定弱权重先验。
    每个约束的罚项直接复用现有的二阶损失，乘子项用对应的一阶残差。

    与论文的两点差异（均为本项目结构所迫，已在 TODO 中记录）：
    1. 论文的 λ 是边界上的逐点场，本项目的两个约束各自已是标量聚合，故 λ 为两个标量。
    2. 多出 w_hist·L_hist 一项统计先验（论文没有对应物），按固定弱权重处理。

    λ 用梯度上升更新：∂L/∂λ_i = r_i，故 λ_i ← λ_i + η_λ · r_i。
    λ 与 β 都是不参与反向传播的 buffer。
    """

    # 约束顺序：0 = 背景，1 = 轮廓占比
    def __init__(self, lambda_ratio):
        super().__init__()
        self.lambda_ratio = lambda_ratio
        self.register_buffer("lambdas", torch.zeros(2))
        self.register_buffer("betas", torch.zeros(2))
        self.register_buffer("lambda_lrs", torch.zeros(2))
        self.register_buffer("w_histogram", torch.zeros(()))

    def total(self, amplitude, histogram, penalties, residuals):
        """penalties = (L_bg, L_area) 二阶罚项；residuals = (r_bg, r_area) 一阶残差。"""
        penalty = (self.betas * torch.stack(penalties)).sum()
        multiplier = (self.lambdas * torch.stack(residuals)).sum()
        return amplitude + self.w_histogram * histogram + penalty + multiplier

    @torch.no_grad()
    def calibrate(self, amplitude, histogram, penalties):
        """首步自动标定 β 与直方图权重，使各项初始量级与振幅项对齐。

        论文的 β 靠网格搜索选取（Table B.5 取 1e2~1e3），本项目改为由第一次前向的
        实测损失自动推出，避免每张图重调——这是唯一保留自 Kendall 方案的部分。
        λ 按论文从 0 起步。
        """
        anchor = amplitude.detach().clamp_min(1e-12)
        self.w_histogram.copy_(anchor / histogram.detach().clamp_min(1e-12))
        initial = torch.stack([penalty.detach() for penalty in penalties])
        self.betas.copy_(anchor / initial.clamp_min(1e-12))
        # 两个约束的 β 量级相差数千倍，故 η_λ 按各自的 β 成比例分配。
        self.lambda_lrs.copy_(self.betas * self.lambda_ratio)
        self.lambdas.zero_()

    @torch.no_grad()
    def update_multipliers(self, residuals):
        """梯度上升：λ_i ← λ_i + η_λ · ∂L/∂λ_i，而 ∂L/∂λ_i = r_i。

        λ 不限号：占比偏小时 r_area < 0 使 λ_area 变负，最小化 λ·r 即把占比推大
        （撑开轮廓）；偏大时反向。故单个乘子天然双侧约束，无需容差。
        r → 0 时 λ 自动停止变化，这是 λ 由违反量驱动、而非由 1/L 驱动的直接体现。
        """
        current = torch.stack([residual.detach() for residual in residuals])
        self.lambdas.add_(self.lambda_lrs * current)

    def state(self):
        return self.lambdas.tolist() + self.betas.tolist()
