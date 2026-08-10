import torch

from losses import (UncertaintyWeights, background_loss,
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


SHARES = (1.0, 0.5, 0.10)
FIRST = (torch.tensor(2e-4), torch.tensor(3e-2), torch.tensor(7e-3))


def make_weights():
    """建好 UncertaintyWeights 并用首轮损失完成归一化基准记录。"""
    weights = UncertaintyWeights(SHARES[1], SHARES[2])
    weights.initialize(*FIRST)
    return weights


def test_initial_contributions_match_shares():
    """s_i 初值为 0、L̂_i,0 = 1，所以首轮三项贡献恰好等于 w_i。"""
    weights = make_weights()
    contributions = weights.contributions(*FIRST)
    assert torch.allclose(contributions, torch.tensor(SHARES), rtol=1e-5)
    # total 首轮 = Σ w_i·(1 + 0) = Σ w_i
    assert torch.allclose(weights.total(*FIRST), torch.tensor(sum(SHARES)), rtol=1e-5)


def test_contributions_converge_to_shares_at_fixed_point():
    """不动点 exp(-s_i) = 1/L̂_i 处，三项贡献仍等于 w_i——与损失降了多少无关。

    这是 w_i 乘在方括号外的关键性质：w_i 若乘进括号内，贡献会收敛到 w_i·L_i,0，
    辅助项被压掉几千倍。
    """
    weights = make_weights()
    drop = torch.tensor([285.9, 82.0, 41.6])          # 各项损失下降倍数（实测量级）
    later = [first / factor for first, factor in zip(FIRST, drop)]
    with torch.no_grad():
        weights.log_variance.copy_(torch.log(1.0 / drop))   # s_i = log(L̂_i)
    contributions = weights.contributions(*later)
    assert torch.allclose(contributions, torch.tensor(SHARES), rtol=1e-4)


def test_log_variance_gradient_vanishes_at_first_step():
    """首轮 s_i = 0、L̂_i = 1 恰好就是不动点，所以 s 的梯度精确为 0。

    ∂L/∂s_i = w_i·(1 - exp(-s_i)·L̂_i)，代入 s_i=0、L̂_i=1 得 0。这不是 bug：
    s 要等损失开始下降（L̂_i < 1）才有驱动力。
    """
    weights = make_weights()
    assert weights.log_variance.requires_grad
    weights.total(*FIRST).backward()
    assert weights.log_variance.grad is not None
    assert torch.allclose(weights.log_variance.grad, torch.zeros(3), atol=1e-7)


def test_log_variance_receives_gradient_once_losses_drop():
    """损失下降后 L̂_i < 1，三个 s_i 都收到正梯度（推 s 往负走、放大权重）。"""
    weights = make_weights()
    later = [loss / 10.0 for loss in FIRST]
    weights.total(*later).backward()
    # ∂L/∂s_i = w_i·(1 - L̂_i) = w_i·0.9 > 0
    expected = torch.tensor(SHARES) * 0.9
    assert torch.allclose(weights.log_variance.grad, expected, rtol=1e-5)


def test_total_goes_negative_at_fixed_point_but_weighted_stays_positive():
    """不动点处 total 因 +s_i 而为负，加权损失仍为正——两者读数含义不同。"""
    weights = make_weights()
    drop = torch.tensor([285.9, 82.0, 41.6])
    later = [loss / factor for loss, factor in zip(FIRST, drop)]
    with torch.no_grad():
        weights.log_variance.copy_(torch.log(1.0 / drop))
    assert bool((weights.contributions(*later) > 0).all())
    # total = Σ w_i·(1 + s_i)，s_i = -log(drop) 远小于 -1，故为负
    expected = (torch.tensor(SHARES) * (1.0 + weights.log_variance)).sum()
    assert torch.allclose(weights.total(*later), expected, rtol=1e-4)
    assert weights.total(*later).item() < 0.0
