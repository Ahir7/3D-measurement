import torch

# Import helper directly to avoid requiring GPU/model
from src.depth.metric3d_gpu import ensure_bchw


def test_ensure_bchw_from_bhw():
    x = torch.randn(2, 512, 512)  # [B, H, W]
    y = ensure_bchw(x)
    assert y.shape == (2, 3, 512, 512)


def test_ensure_bchw_from_hw():
    x = torch.randn(256, 320)  # [H, W]
    y = ensure_bchw(x)
    assert y.shape == (1, 3, 256, 320)


def test_ensure_bchw_from_chw_rgb():
    x = torch.randn(3, 480, 640)  # [C, H, W]
    y = ensure_bchw(x)
    assert y.shape == (1, 3, 480, 640)


def test_ensure_bchw_from_chw_gray():
    x = torch.randn(1, 128, 128)  # [C, H, W]
    y = ensure_bchw(x)
    assert y.shape == (1, 3, 128, 128)


def test_ensure_bchw_from_bhwc():
    x = torch.randn(4, 300, 200, 3)  # [B, H, W, 3]
    y = ensure_bchw(x)
    assert y.shape == (4, 3, 300, 200)


def test_ensure_bchw_from_bchw_gray():
    x = torch.randn(5, 1, 64, 64)  # [B, 1, H, W]
    y = ensure_bchw(x)
    assert y.shape == (5, 3, 64, 64)


