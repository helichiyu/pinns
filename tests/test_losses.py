import torch

from losses import (CalibratedWeights, background_loss,
                    masked_histogram_quantile_loss, quantile_ranks,
                    source_contour, topk_contour)

SIGMA = 4.0
THRESHOLD = 0.20
BINS = 64


def make_disk(size=64, radius=14):
    """造一张合成图：暗背景上一个亮圆盘，亮度从中心向外递减。"""
    axis = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    distance = (axis[:, None].square() + axis[None, :].square()).sqrt()
    disk = (1.0 - distance / radius).clamp_min(0.0)
    return disk[None, None]


def test_source_contour_is_binary():
    """源图轮廓只能取 0 或 1。"""
    contour = source_contour(make_disk(), SIGMA, THRESHOLD)
    assert torch.equal(contour, contour.round())
    assert contour.amin() == 0.0
    assert contour.amax() == 1.0


def test_topk_contour_has_exactly_k_pixels():
    """输出轮廓的 1 像素数恰好等于 k。"""
    prediction = torch.rand(1, 1, 64, 64)
    for k in (1, 37, 500):
        contour = topk_contour(prediction, SIGMA, k)
        assert int(contour.sum().item()) == k


def test_topk_contour_selects_brightest():
    """选中的必须是模糊图里最亮的那批像素：mask 内的最小值不低于 mask 外的最大值。"""
    from losses import gaussian_blur
    prediction = torch.rand(1, 1, 64, 64)
    blurred = gaussian_blur(prediction, SIGMA)
    contour = topk_contour(prediction, SIGMA, 300)
    inside = blurred[contour > 0.5]
    outside = blurred[contour < 0.5]
    assert inside.amin() >= outside.amax()


def test_masks_have_equal_pixel_count():
    """源图 mask 与输出 mask 的像素数严格相等——这是轮廓内直方图不需要长度补偿的前提。"""
    source = make_disk()
    source_mask = source_contour(source, SIGMA, THRESHOLD)
    k = int(source_mask.sum().item())
    prediction_mask = topk_contour(torch.rand_like(source), SIGMA, k)
    assert int(source_mask.sum().item()) == int(prediction_mask.sum().item())


def test_topk_contour_is_constant():
    """mask 是纯常量，不参与求导。"""
    prediction = torch.rand(1, 1, 64, 64, requires_grad=True)
    contour = topk_contour(prediction, SIGMA, 200)
    assert contour.requires_grad is False


def make_target_quantiles(source, mask, ranks):
    """从源图 mask 内取 B 个分位值，与 train.py 的预计算同形。"""
    return source[mask > 0.5].sort().values[ranks]


def test_quantile_ranks_are_valid():
    """rank 下标必须严格递增、落在 [0, k) 内，且末位恰为 k-1（覆盖最亮像素）。"""
    for k in (100, 577, 1024):
        ranks = quantile_ranks(k, BINS, torch.device("cpu"))
        assert ranks.numel() == BINS
        assert ranks[0].item() >= 0
        assert ranks[-1].item() == k - 1
        assert bool((ranks.diff() > 0).all())


def test_gradient_flows_to_prediction():
    """mask 不可导，但梯度必须经 prediction 回流到网络。"""
    source = make_disk()
    source_mask = source_contour(source, SIGMA, THRESHOLD)
    k = int(source_mask.sum().item())
    ranks = quantile_ranks(k, BINS, source.device)
    target_quantiles = make_target_quantiles(source, source_mask, ranks)

    prediction = torch.rand_like(source).requires_grad_(True)
    contour = topk_contour(prediction, SIGMA, k)
    loss = (background_loss(prediction, contour)
            + masked_histogram_quantile_loss(prediction, contour, target_quantiles, ranks))
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad.norm().item() > 0.0


def test_quantile_gradient_hits_exactly_bins_pixels():
    """等点数分箱每轮只有 B 个像素直接收到直方图梯度（其余靠振幅与背景间接训练）。"""
    source = make_disk()
    source_mask = source_contour(source, SIGMA, THRESHOLD)
    k = int(source_mask.sum().item())
    ranks = quantile_ranks(k, BINS, source.device)
    target_quantiles = make_target_quantiles(source, source_mask, ranks)

    prediction = torch.rand_like(source).requires_grad_(True)
    contour = topk_contour(prediction, SIGMA, k)
    masked_histogram_quantile_loss(prediction, contour, target_quantiles, ranks).backward()
    assert int((prediction.grad != 0).sum().item()) == BINS


def test_masked_histogram_zero_for_identical_distribution():
    """同一张图与自己比，排序后逐位相等，分位值损失严格为 0（无浮点残差）。"""
    source = make_disk()
    mask = source_contour(source, SIGMA, THRESHOLD)
    ranks = quantile_ranks(int(mask.sum().item()), BINS, source.device)
    target_quantiles = make_target_quantiles(source, mask, ranks)
    loss = masked_histogram_quantile_loss(source, mask, target_quantiles, ranks)
    assert loss.item() == 0.0


def test_background_loss_zero_outside_contour():
    """轮廓外全为 0 时背景损失为 0；轮廓外有亮度时严格为正。"""
    contour = torch.zeros(1, 1, 8, 8)
    contour[..., 2:6, 2:6] = 1.0
    clean = contour.clone()
    assert background_loss(clean, contour).item() == 0.0
    dirty = clean.clone()
    dirty[..., 0, 0] = 0.5
    assert background_loss(dirty, contour).item() > 0.0


def test_calibrate_makes_contributions_match_shares():
    """标定后，两项的加权贡献应恰好等于 SHARE × 振幅损失。"""
    weights = CalibratedWeights(0.5, 0.10)
    amplitude = torch.tensor(2e-4)
    histogram = torch.tensor(3e-2)
    background = torch.tensor(7e-3)
    weights.calibrate(amplitude, histogram, background)
    contributions = weights.betas * torch.stack([histogram, background])
    assert torch.allclose(contributions, amplitude * torch.tensor([0.5, 0.10]), rtol=1e-5)
    total = weights.total(amplitude, histogram, background)
    assert torch.allclose(total, amplitude * 1.60, rtol=1e-5)
