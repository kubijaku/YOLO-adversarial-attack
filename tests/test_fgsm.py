from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from attack.fgsm import (
    compute_proxy_from_preds,
    find_images_recursive,
    flatten_pred_tensor,
    load_image_tensor,
    save_tensor_image,
)


# -------------------------
# Helpers
# -------------------------
@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    path = tmp_path / "test.png"
    img.save(path)
    return path


@pytest.fixture
def device() -> str:
    return "cpu"


# -------------------------
# find_images_recursive
# -------------------------
def test_find_images_recursive(tmp_path: Path):
    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.png"
    txt = tmp_path / "c.txt"

    Image.new("RGB", (10, 10)).save(img1)
    Image.new("RGB", (10, 10)).save(img2)
    txt.write_text("ignore me")

    files = find_images_recursive(tmp_path)

    assert len(files) == 2
    assert files[0].endswith(".jpg")
    assert files[1].endswith(".png")


def test_find_images_recursive_nonexistent():
    files = find_images_recursive("this/path/does/not/exist")
    assert files == []


# -------------------------
# load_image_tensor
# -------------------------
def test_load_image_tensor(sample_image: Path, device: str):
    tensor, arr = load_image_tensor(sample_image, device)

    assert isinstance(tensor, torch.Tensor)
    assert isinstance(arr, np.ndarray)

    assert tensor.shape == (1, 3, 32, 32)
    assert arr.shape == (32, 32, 3)
    assert tensor.device.type == "cpu"


# -------------------------
# save_tensor_image
# -------------------------
def test_save_tensor_image(tmp_path: Path):
    tensor = torch.rand(1, 3, 16, 16)
    out_path = tmp_path / "out.png"

    save_tensor_image(tensor, out_path)

    assert out_path.exists()
    img = Image.open(out_path)
    assert img.size == (16, 16)


# -------------------------
# flatten_pred_tensor
# -------------------------
def test_flatten_pred_tensor_4d():
    p = torch.randn(2, 10, 4, 4)
    flat = flatten_pred_tensor(p)

    assert flat.shape == (2, 16, 10)


def test_flatten_pred_tensor_invalid_type():
    with pytest.raises(TypeError):
        flatten_pred_tensor("not a tensor")


# -------------------------
# compute_proxy_from_preds
# -------------------------
def test_compute_proxy_from_preds_no_gt():
    pred = torch.randn(1, 10, 4, 4)
    loss = compute_proxy_from_preds(pred, device="cpu")

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_compute_proxy_from_preds_with_gt():
    pred = torch.randn(1, 10, 4, 4)
    gt_boxes = [(0, 0.5, 0.5, 0.2, 0.2)]

    loss = compute_proxy_from_preds(pred, device="cpu", gt_boxes=gt_boxes)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
