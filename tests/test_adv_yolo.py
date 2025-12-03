import numpy as np
import pytest
import torch
from PIL import Image

from attack.fgsm import *


# Adjust this import if your module filename is different


def test_find_images_recursive_and_suffixes(tmp_path):
    # create some files
    d = tmp_path / "images"
    d.mkdir()
    (d / "a.jpg").write_bytes(b"dummy")
    (d / "b.png").write_bytes(b"dummy")
    (d / "c.txt").write_text("not an image")
    # nested
    nested = d / "sub"
    nested.mkdir()
    (nested / "d.jpeg").write_bytes(b"dummy")

    found = find_images_recursive(d)
    # sorted list of paths
    assert all(isinstance(p, str) for p in found)
    # should include only supported suffixes and 3 images
    assert len(found) == 3
    assert any("a.jpg" in p for p in found)
    assert any("b.png" in p for p in found)
    assert any("d.jpeg" in p for p in found)


def test_read_yolo_label_file(tmp_path):
    # empty -> no file
    nofile = tmp_path / "no.txt"
    assert read_yolo_label_file(str(nofile)) == []

    # create valid file
    f = tmp_path / "labels.txt"
    f.write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05\ninvalid line here\n")
    boxes = read_yolo_label_file(str(f))
    assert len(boxes) == 2
    assert boxes[0][0] == 0
    assert pytest.approx(boxes[0][1]) == 0.5

    # numeric class can be float formatted
    f2 = tmp_path / "labels2.txt"
    f2.write_text("2.0 0.2 0.2 0.1 0.1\n")
    boxes2 = read_yolo_label_file(str(f2))
    assert boxes2[0][0] == 2


def test_load_and_save_image_tensor_roundtrip(tmp_path):
    # create a small RGB image
    img_path = tmp_path / "img.png"
    arr = (np.random.rand(16, 16, 3) * 255).astype("uint8")
    Image.fromarray(arr).save(img_path)

    device = torch.device("cpu")
    t, arr_f = load_image_tensor(str(img_path), device)
    assert isinstance(t, torch.Tensor)
    assert t.shape[0] == 1 and t.shape[1] == 3

    out_path = tmp_path / "out.png"
    # produce a tensor in same shape expected by save_tensor_image:
    # (1, C, H, W) or any tensor convertible in function
    save_tensor_image(t, str(out_path))
    assert out_path.exists()
    saved = Image.open(out_path)
    assert saved.mode == "RGB"
    assert saved.size == (16, 16)


def test_flatten_pred_tensor_variants():
    # Case (B,C,H,W)
    B, C, H, W = 2, 8, 4, 5
    t = torch.arange(B * C * H * W, dtype=torch.float32).reshape(B, C, H, W)
    flat = flatten_pred_tensor(t)
    assert flat.shape == (B, H * W, C)

    # dims <4 fallback: e.g., (B, N, C) -> should keep structure or reshape accordingly
    t2 = torch.randn(2, 10, 6)
    flat2 = flatten_pred_tensor(t2)
    # function expects last dim as channels in fallback; for (B,N,C) will return (B, N, C)
    assert flat2.shape == (2, 10, 6)

    # Non-tensor raises TypeError
    with pytest.raises(TypeError):
        flatten_pred_tensor([1, 2, 3])


def test_compute_proxy_from_preds_no_gt():
    # create a simple prediction: list of 1 tensor shaped (B,C,H,W) with C>5
    B, C, H, W = 1, 6, 3, 3
    # create a tensor that depends on input for autograd is not necessary here; just test output shape/number
    t = torch.randn(B, C, H, W, requires_grad=False)
    loss = compute_proxy_from_preds([t])
    # no gt: should be scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0 or loss.numel() == 1

    # With gt_boxes: ensure it returns scalar
    gt = [(0, 0.5, 0.5, 0.1, 0.1)]
    loss2 = compute_proxy_from_preds([t], gt_boxes=gt)
    assert isinstance(loss2, torch.Tensor)


def test_compute_proxy_from_preds_with_tensor_graph():
    # Build a tensor that is a function of an input so loss.backward() will produce gradient on input.
    # We'll simulate a predicted tensor computed from an input image tensor to preserve autograd.
    img = torch.rand(1, 3, 8, 8, requires_grad=True)
    # internal model: produce 1 head with channels >5, e.g., C=6
    # make it a simple linear op on img that keeps grad path
    head = torch.cat([img, img.mean(dim=(2, 3), keepdim=True).expand(1, 3, 8, 8)], dim=1)  # now C=6
    # ensure head requires grad
    head = head * 1.0
    # compute loss using compute_proxy_from_preds: it expects (B,C,H,W)
    loss = compute_proxy_from_preds([head], gt_boxes=None)
    # loss should be differentiable w.r.t. head (and hence img if head depends on img)
    loss.backward()
    # Since head derived from img, img should have grad
    assert img.grad is not None
    assert img.grad.abs().sum() > 0


