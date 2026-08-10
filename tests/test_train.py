import torch

from losses import amplitude_mask
from train import patterson_initialization

DC_RADIUS = 3.0


def make_disk(size=64, radius=14):
    """造一张合成图：暗背景上一个亮圆盘，亮度从中心向外递减。"""
    axis = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    distance = (axis[:, None].square() + axis[None, :].square()).sqrt()
    disk = (1.0 - distance / radius).clamp_min(0.0)
    return disk[None, None]


def make_patterson(source):
    """按 train.py 的调用方式构造 Patterson 输入。"""
    amplitude = torch.abs(torch.fft.fft2(source)).detach()
    return patterson_initialization(amplitude, amplitude_mask(amplitude, DC_RADIUS))


def test_patterson_is_normalized_to_unit_range():
    """min-max 归一化后取值恰好铺满 [0, 1]，UNet 输入无归一化层，范围必须可控。"""
    patterson = make_patterson(make_disk())
    assert patterson.amin().item() == 0.0
    assert patterson.amax().item() == 1.0
    assert bool(torch.isfinite(patterson).all())


def test_patterson_matches_source_shape():
    """输入与源图同形，才能直接喂进 UNet 并让输出与源图逐点比较。"""
    for height, width in ((64, 64), (48, 80)):
        source = torch.rand(1, 1, height, width)
        assert make_patterson(source).shape == source.shape


def test_patterson_peak_sits_at_canvas_center():
    """fftshift 后自相关原点必须落在画布中心，与源图物体居中、后续轮次输入居中一致。"""
    patterson = make_patterson(make_disk())
    height, width = patterson.shape[-2:]
    assert divmod(patterson.argmax().item(), width) == (height // 2, width // 2)


def test_patterson_is_centrosymmetric():
    """自相关恒有 P(u) = P(-u)。下标 0 那一行/列没有对称伙伴，排除后比较。"""
    core = make_patterson(make_disk())[..., 1:, 1:]
    assert torch.allclose(core, torch.flip(core, dims=(-2, -1)), atol=1e-6)


def test_patterson_is_deterministic():
    """Patterson 由实测振幅唯一确定，不含随机性——seed 从此只影响网络权重初始化。"""
    source = make_disk()
    assert torch.equal(make_patterson(source), make_patterson(source))


def test_patterson_excludes_masked_frequencies():
    """挖洞真的生效：低频圆盘参与与否，结果必须不同（否则 mask 传参形同摆设）。"""
    source = make_disk()
    amplitude = torch.abs(torch.fft.fft2(source)).detach()
    holed = patterson_initialization(amplitude, amplitude_mask(amplitude, DC_RADIUS))
    intact = patterson_initialization(amplitude, torch.ones_like(amplitude, dtype=torch.bool))
    assert not torch.allclose(holed, intact, atol=1e-3)
