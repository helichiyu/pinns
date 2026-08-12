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


def amplitude_mask(target_amplitude, radius):
    """挖掉以直流为中心的低频圆盘，其余频点（含高频）全部参与损失。

    直流点等于全图像素总和，紧邻它的几圈低频同样是整幅谱里数值最大的一批，留着
    损失会几乎只在调整整体亮度和大尺度明暗。高频虽然幅值小、含噪，但决定重建的
    锐度，不排除。

    fft2 没有 fftshift，直流在 [0, 0]、低频绕到四个角，所以频率距离要取环绕后的
    最小值，不能直接切一块方块。radius=3.0 时挖掉 29 个点。

    只在算损失时挖：训练过程本身用的仍是完整的 prediction，没有任何频域裁剪。
    """
    height, width = target_amplitude.shape[-2:]
    device = target_amplitude.device
    offset_y = torch.arange(height, device=device)
    offset_y = torch.minimum(offset_y, height - offset_y)
    offset_x = torch.arange(width, device=device)
    offset_x = torch.minimum(offset_x, width - offset_x)
    distance = offset_y[:, None].square() + offset_x[None, :].square()
    return (distance > radius**2).expand_as(target_amplitude)


def normalized_log_amplitude_loss(prediction, target_amplitude, valid_mask):
    """归一化基准取挖洞后的最大振幅，即参与损失的频点里最大的那个。

    用被挖掉的直流当基准会让剩下的频点全挤在 log1p 的近零段，动态范围浪费掉。
    """
    prediction_amplitude = torch.abs(torch.fft.fft2(prediction))
    scale = target_amplitude[valid_mask].amax().detach().clamp_min(1e-8)
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


def quantile_ranks(pixels, bins, device):
    """等点数分箱的 rank 下标：把 k 个排序后的像素切成 bins 段，取每段末位。

    整数运算，训练前算一次复用。要求 bins ≤ pixels，否则首个下标为负、会静默
    绕回末位，拿到错的分位值——调用方须先校验。
    """
    return (torch.arange(1, bins + 1, device=device) * pixels // bins) - 1


def masked_histogram_quantile_loss(prediction, prediction_mask, target_quantiles, ranks):
    """只在轮廓内比较分布：等点数分箱，比 B 个分位点的像素值。

    每箱点数相等，所以能比的只有箱边界（分位值）——比箱内计数是常数，没意义。
    预测分布取自输出自己的 mask，而不是源图 mask——后者等于把源图轮廓的位置
    当成监督。target_quantiles 由调用方用源图 mask 预先算好，每轮复用。

    sort 与固定下标 gather 都原生可导：排在第 j 名的像素领走 sorted[j] 的梯度，
    所以每轮恰有 B 个像素直接收到直方图梯度。
    """
    values = prediction[prediction_mask > 0.5]
    return F.mse_loss(values.sort().values[ranks], target_quantiles)


def background_loss(prediction, contour):
    """轮廓外的输出被约束为 0。contour 已是常量，无需再 detach。"""
    return torch.mean(((1.0 - contour) * prediction).square())


def input_output_loss(prediction, model_input):
    """Penalize changes between one iteration's input and output."""
    return F.mse_loss(prediction, model_input)


class UncertaintyWeights(nn.Module):
    """Kendall 同方差不确定性加权，外层再乘固定的目标贡献比。

        L = Σ_i w_i · [ exp(-s_i) · L̂_i + s_i ]，   L̂_i = L_i / L_i,0

    `s_i = log σ_i²` 是可学习参数（Kendall 2017，arXiv:1705.07115），与网络参数
    进同一个 Adam。`exp(-s_i)` 是自适应权重，`+ s_i` 是阻止权重趋零的正则项。

    两处设计与原式不同，各有其必要性：

    1. **各项先除以首轮实测值 L_i,0**。三项损失的原始量级差近四千倍（振幅在
       log1p 压缩后的 18 万个频点上求 MSE，天然是 1e-5；直方图是 100 个像素值的
       MSE；背景是全图 pred² 的均值），不归一化则 `s_i = 0` 的开局会把辅助项放大
       数千倍，而全黑解能让背景损失精确归零 —— 大概率直接塌缩。归一化后 L̂_i,0 = 1，
       `s_i = 0` 正是原文初值，无需额外的初始化步骤。

    2. **w_i 乘在方括号外**。对 s_i 求导时 w_i 被约掉，不动点仍是 exp(-s_i) = 1/L̂_i，
       但各项的实际加权贡献收敛到 w_i 本身，即全程维持 1 : w_hist : w_bg，而不是
       原式的三项等权。w_i 若乘进方括号内（作用在 L̂_i 上），不动点会让贡献收敛到
       w_i·L_i,0 —— 辅助项被压掉几千倍，退化成"振幅独占"，即 docs 第 6.2 节记录的
       旧失败形态。

    与先前 CalibratedWeights 的关系：后者的 β_i = SHARE_i·L_amp,0/L_i,0，与本式首轮
    的有效权重 w_i/L_i,0 只差一个全局公因子 L_amp,0，而 Adam 对损失的全局缩放不敏感
    （m/√v 归一化）。所以第一轮的网络更新与固定权重方案一致，差异纯粹来自 s 可学。
    """

    def __init__(self, share_histogram, share_background, share_input_output=0.0):
        super().__init__()
        # Keep this order consistent across training, history, and frontend metrics.
        self.log_variance = nn.Parameter(torch.zeros(4))
        self.register_buffer("shares", torch.tensor(
            [1.0, share_histogram, share_background, share_input_output]))
        self.register_buffer("initial", torch.ones(4))

    @torch.no_grad()
    def initialize(self, amplitude, histogram, background, input_output):
        """Record first-step losses as normalization baselines."""
        losses = torch.stack([loss.detach() for loss in
                              (amplitude, histogram, background, input_output)])
        self.initial.copy_(losses.clamp_min(1e-12))

    def total(self, amplitude, histogram, background, input_output):
        """被最小化的目标。含 +s_i 正则项，s_i 变负时会小于 0。"""
        contributions = self.contributions(amplitude, histogram, background, input_output)
        return (contributions + self.shares * self.log_variance).sum()

    def contributions(self, amplitude, histogram, background, input_output):
        """三项的实际加权贡献 w_i·exp(-s_i)·L̂_i，收敛目标是 w_i 本身。

        它们的和是"加权损失"：不含 +s_i，所以只反映损失下降、不反映 s 走了多远，
        量级也与固定权重方案的 total 可比。total 变小可能只是 s 在适配。
        """
        normalized = torch.stack([amplitude, histogram, background, input_output]) / self.initial
        return self.shares * torch.exp(-self.log_variance) * normalized
