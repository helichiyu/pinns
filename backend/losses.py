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


def source_contour(source, sigma, threshold):
    """源图硬轮廓，整个训练只算一次。

    模糊 → 减去自身最低基线 → 峰值归一化 → 硬阈值。减基线是为了避免整体亮度
    把轮廓撑到全图；阈值是相对峰值的比例，所以不随图片整体亮度漂移。
    """
    blurred = gaussian_blur(source, sigma)
    contrast = blurred - blurred.amin()
    return (contrast / contrast.amax().clamp_min(1e-6) > threshold).to(source.dtype)


def topk_contour(prediction, sigma, k):
    """输出侧硬轮廓：模糊后取前 k 个最亮像素置 1。

    像素数恒为 k，与源图轮廓严格相等，所以轮廓内直方图两侧长度天然一致。
    用 topk 的下标散射而不是阈值比较，是因为并列值在阈值下会多圈几个像素。

    mask 的 requires_grad 为 False，是纯常量，只当选择器用——它本身不可导也不
    需要可导，梯度经 prediction 回流到网络。
    """
    blurred = gaussian_blur(prediction, sigma).flatten()
    mask = torch.zeros_like(blurred)
    mask[torch.topk(blurred, k).indices] = 1.0
    return mask.view_as(prediction)


def soft_cdf(values, bins, softness):
    """一组像素值的可微软 CDF，长度为 bins。"""
    centers = torch.linspace(0, 1, bins, device=values.device, dtype=values.dtype)
    return torch.sigmoid((centers[:, None] - values[None, :]) / softness).mean(dim=1)


def masked_histogram_cdf_loss(prediction, prediction_mask, target_cdf, bins, softness):
    """只在轮廓内比较分布。

    预测分布取自输出自己的 mask，而不是源图 mask——后者等于把源图轮廓的位置
    当成监督。target_cdf 由调用方用源图 mask 预先算好，每轮复用。
    """
    values = prediction[prediction_mask > 0.5]
    return F.mse_loss(soft_cdf(values, bins, softness), target_cdf)


def background_loss(prediction, contour):
    """轮廓外的输出被约束为 0。contour 已是常量，无需再 detach。"""
    return torch.mean(((1.0 - contour) * prediction).square())


class CalibratedWeights(nn.Module):
    """固定权重的损失组合，权重由首步前向自动标定。

        L = L_amp + β_hist·L_hist + β_bg·L_bg

    振幅是唯一的硬数据，权重恒为 1。另两项是静态罚项，β 在第一次前向定一次，
    使各项的初始加权贡献等于 SHARE_i × L_amp,0。SHARE_i 按各约束的可信度排定
    （振幅 > 轮廓内直方图 > 背景），是全局常数而非每图超参：换图片时各 L_i,0
    都变、β 跟着变，但贡献占比不变。

    没有任何一项带自适应拉格朗日乘子。早先给占比项配对偶变量做梯度上升，收敛
    末期 θ 与 λ 互相追逐形成极限环，损失上下振荡；静态罚项没有这个反馈回路。
    """

    def __init__(self, share_histogram, share_background):
        super().__init__()
        self.shares = (share_histogram, share_background)
        self.register_buffer("betas", torch.zeros(2))   # 顺序：直方图、背景

    def total(self, amplitude, histogram, background):
        weighted = self.betas * torch.stack([histogram, background])
        return amplitude + weighted.sum()

    @torch.no_grad()
    def calibrate(self, amplitude, histogram, background):
        """首步标定：β_i = SHARE_i × L_amp,0 / L_i,0。"""
        anchor = amplitude.detach().clamp_min(1e-12)
        initial = torch.stack([loss.detach() for loss in (histogram, background)])
        shares = torch.tensor(self.shares, device=initial.device, dtype=initial.dtype)
        self.betas.copy_(shares * anchor / initial.clamp_min(1e-12))
