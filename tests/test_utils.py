import os
from pathlib import Path

import numpy as np
import pytest

from utils.utils import (
    compute_iou_matrix,
    get_label_file,
    get_matched_ground_truth_and_predictions,
    get_yolo_boxes,
    iou_xyxy,
    load_images,
    normalize_confusion_matrix,
    yolo_norm_to_xyxy,
)


# -------------------------
# get_yolo_boxes
# -------------------------
def test_get_yolo_boxes(tmp_path: Path):
    label = tmp_path / "label.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.3 0.3")

    boxes = get_yolo_boxes(label)

    assert len(boxes) == 2
    assert boxes[0] == (0, 0.5, 0.5, 0.2, 0.2)


def test_get_yolo_boxes_missing():
    assert get_yolo_boxes("missing.txt") == []


# -------------------------
# yolo_norm_to_xyxy
# -------------------------
def test_yolo_norm_to_xyxy():
    xyxy = yolo_norm_to_xyxy(xc=0.5, yc=0.5, w=0.2, h=0.4, img_w=100, img_h=200)

    assert np.allclose(xyxy, [40.0, 60.0, 60.0, 140.0])


# -------------------------
# iou_xyxy
# -------------------------
def test_iou_xyxy_identical_boxes():
    box = [10, 10, 20, 20]
    assert iou_xyxy(box, box) == 1.0


def test_iou_xyxy_no_overlap():
    boxA = [0, 0, 10, 10]
    boxB = [20, 20, 30, 30]
    assert iou_xyxy(boxA, boxB) == 0.0


# -------------------------
# load_images
# -------------------------
def test_load_images(tmp_path: Path):
    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.png"
    txt = tmp_path / "c.txt"

    img1.touch()
    img2.touch()
    txt.touch()

    images = load_images(tmp_path)

    assert len(images) == 2
    assert images[0].endswith(".jpg")
    assert images[1].endswith(".png")


# -------------------------
# get_label_file
# -------------------------
def test_get_label_file(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    label = labels_dir / "image_001.txt"
    label.write_text("0 0.5 0.5 0.2 0.2")

    img_path = tmp_path / "image_001.jpg"

    found = get_label_file(str(labels_dir), str(img_path))
    assert found.endswith("image_001.txt")


def test_get_label_file_missing(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        get_label_file(str(labels_dir), "missing_image.jpg")


# -------------------------
# compute_iou_matrix
# -------------------------
def test_compute_iou_matrix():
    gt = [{"xyxy": [0, 0, 10, 10]}]
    preds = [{"xyxy": [5, 5, 15, 15]}]

    iou_matrix = compute_iou_matrix(gt, preds)

    assert iou_matrix.shape == (1, 1)
    assert 0.0 < iou_matrix[0, 0] < 1.0


# -------------------------
# get_matched_ground_truth_and_predictions
# -------------------------
def test_matching_logic():
    gt_objects = [{"cls": 0, "xyxy": [0, 0, 10, 10]}]
    predictions = [{"cls": 0, "xyxy": [0, 0, 10, 10]}]

    cm = np.zeros((2, 2), dtype=int)
    iou_matrix = compute_iou_matrix(gt_objects, predictions)

    matched_gt, matched_pred = get_matched_ground_truth_and_predictions(
        gt_objects=gt_objects,
        predictions=predictions,
        iou_matrix=iou_matrix,
        iou_threshold=0.5,
        confusion_matrix=cm,
    )

    assert matched_gt == {0}
    assert matched_pred == {0}
    assert cm[0, 0] == 1


# -------------------------
# normalize_confusion_matrix
# -------------------------
def test_normalize_confusion_matrix():
    cm = np.array([[2, 2], [0, 0]])
    norm = normalize_confusion_matrix(cm)

    assert np.allclose(norm[0], [0.5, 0.5])
    assert np.all(norm[1] == 0.0)
