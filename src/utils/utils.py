# script generate confusion matrix's to enable easy comparison of the results for the model
# for validation data with new created adversarial images
import os
from typing import Any

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ---------------- util functions ----------------
def get_yolo_boxes(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
                boxes.append((cls, xc, yc, w, h))
    return boxes


def yolo_norm_to_xyxy(xc, yc, w, h, img_w, img_h):
    x_c = xc * img_w
    y_c = yc * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = x_c - bw / 2
    y1 = y_c - bh / 2
    x2 = x_c + bw / 2
    y2 = y_c + bh / 2
    return [x1, y1, x2, y2]


def iou_xyxy(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    if union <= 0:
        return 0.0
    return interArea / union


def load_images(folder):
    p = Path(folder)
    imgs = sorted(
        [
            str(f)
            for f in p.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        ]
    )
    return imgs


def get_label_file(labels_dir: str, img_path: str) -> str:
    """
    Function attempts to find corresponding label file by matching basename prefix
    :param labels_dir:
    :param img_path:
    :return: path to label file
    """
    label_file = None
    for f in Path(labels_dir).rglob("*.txt"):
        if Path(f).stem in Path(img_path).stem:
            label_file = str(f)
            break
    if label_file is None:
        print(f"Warning: no label for {img_path} - skipping")
        raise FileNotFoundError(f"Could not find label file for {img_path}")
    return label_file


def compute_iou_matrix(gt_objects: list, predictions: list) -> np.ndarray:
    """
    Computes IoU matrix of gt_objects and predictions - IoU matrix is a 2D matrix (table) that stores Intersection-over-Union values between every ground-truth box and every predicted box.
    :param gt_objects: list of ground truth boxes
    :param predictions: list of predicted boxes
    """

    iou_matrix = np.zeros((len(gt_objects), len(predictions)), dtype=float)
    for gt_idx in range(len(gt_objects)):
        for pred_idx in range(len(predictions)):
            iou_matrix[gt_idx, pred_idx] = iou_xyxy(
                gt_objects[gt_idx]["xyxy"], predictions[pred_idx]["xyxy"]
            )
    return iou_matrix


def get_matched_ground_truth_and_predictions(
    gt_objects: list,
    predictions: list,
    iou_matrix: np.ndarray,
    iou_threshold: float,
    confusion_matrix: np.ndarray,
):
    matched_ground_truth = set()
    matched_pred = set()

    while True:
        if iou_matrix.size == 0:
            break
        gt_idx, pred_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        max_iou = iou_matrix[gt_idx, pred_idx]
        if max_iou < iou_threshold:
            break
        ground_truth_cls = gt_objects[gt_idx]["cls"]
        pred_cls = predictions[pred_idx]["cls"]
        confusion_matrix[ground_truth_cls, pred_cls] += 1
        matched_ground_truth.add(gt_idx)
        matched_pred.add(pred_idx)
        iou_matrix[gt_idx, :] = -1.0
        iou_matrix[:, pred_idx] = -1.0

    return matched_ground_truth, matched_pred


# ---------------- core: evaluate one folder -> return local confusion_matrix ----------------
def get_gt_objects(
    label_file: str, img_path: str, img_w: int, img_h: int
) -> type[list]:
    gt_objects = []
    if label_file:
        gt_yolo_boxes = get_yolo_boxes(label_file)
        for cls, xc, yc, w, h in gt_yolo_boxes:
            gt_objects.append(
                {
                    "cls": int(cls),
                    "xyxy": yolo_norm_to_xyxy(xc, yc, w, h, img_w, img_h),
                }
            )
    else:
        print("Warning: no gt_objects label found for", img_path, "- skipping")
        raise LookupError("No gt_objects label found for")
    return gt_objects


def get_predictions(results: list, conf_threshold: float, img_path: str) -> list[Any]:
    r = results[0]

    predictions = []

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    detected_class_names = r.boxes.cls.cpu().numpy().astype(int)

    for b, c, cf in zip(boxes_xyxy, detected_class_names, confs):
        if cf < conf_threshold:
            continue
        predictions.append(
            {
                "cls": int(c),
                "xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "conf": float(cf),
            }
        )
    if len(predictions) == 0:
        raise LookupError("No predictions found for", img_path)

    return predictions


def evaluate_confusion_matrix(
    images_dir: str,
    labels_dir: str,
    class_names: list,
    model_path: str,
    device: str,
    conf_threshold: float,
    iou_threshold: float,
):
    """
    Evaluate detections on images in images_dir using gt_objects labels in labels_dir.
    Returns a NEW confusion matrix of shape (num_classes+1, num_classes+1).
    """
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

    print("Loading model for inference:", model_path)
    model = YOLO(model_path)
    model.to(device)

    images = load_images(images_dir)
    print(f"Found {len(images)} images in {images_dir}")

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        try:
            label_file = get_label_file(labels_dir=labels_dir, img_path=img_path)
        except FileNotFoundError:
            print(f"Warning: no label for {img_path} - skipping")
            continue

        gt_objects = get_gt_objects(
            label_file=label_file, img_path=img_path, img_w=img_w, img_h=img_h
        )

        results = model.predict(
            source=img_path, imgsz=max(img_w, img_h), conf=conf_threshold, device=device
        )

        predictions = get_predictions(
            results=results, conf_threshold=conf_threshold, img_path=img_path
        )
        iou_matrix = compute_iou_matrix(gt_objects, predictions)

        matched_ground_truth, matched_pred = get_matched_ground_truth_and_predictions(
            gt_objects=gt_objects,
            predictions=predictions,
            iou_matrix=iou_matrix,
            iou_threshold=iou_threshold,
            confusion_matrix=confusion_matrix,
        )

        for gt_idx in range(len(gt_objects)):
            if gt_idx not in matched_ground_truth:
                confusion_matrix[gt_objects[gt_idx]["cls"], num_classes] += 1

        for pred_idx in range(len(predictions)):
            if pred_idx not in matched_pred:
                confusion_matrix[num_classes, predictions[pred_idx]["cls"]] += 1

    return confusion_matrix


# ---------------- plotting / normalization ----------------
def normalize_confusion_matrix(confusion_matrix):
    with np.errstate(all="ignore"):
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        normalized = confusion_matrix.astype(float) / row_sums
        normalized[np.isnan(normalized)] = 0.0
    return normalized


def plot_and_save_confusion_matrix(
    confusion_matrix, names, out_path="confusion_matrix.png", title="Confusion Matrix"
):
    labels = names + ["No Detection"]
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(confusion_matrix, interpolation="nearest", cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    is_float = np.issubdtype(confusion_matrix.dtype, np.floating)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            value = confusion_matrix[i, j]
            text = f"{value:.2f}" if is_float else f"{int(value)}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > confusion_matrix.max() / 2 else "black",
            )
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print("Saved confusion matrix image to:", out_path)


def save_confusion_matrix_as_csv(
    confusion_matrix: np.ndarray, csv_filename: str, output_dir: str
):
    file_path = os.path.join(output_dir, csv_filename)
    np.savetxt(
        file_path,
        confusion_matrix,
        fmt="%d",
        delimiter=",",
    )
    print("Saved CSVs to", file_path)


def train_model(
    model_path: str = "../model/yolov8n.pt",
    data_yaml: str = "./car/data.yaml",
    device: str = "mps",
    epochs: int = 30,
    batch: int = 10,
) -> dict:
    """
    Function to train the YOLO model
    :param epochs:
    :param batch:
    :param model_path:
    :param data_yaml:
    :param device:
    :return: directory
    """
    model = YOLO(model_path)

    # Training The Final Model
    final_model = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        optimizer="auto",
        device=device,
        project=os.path.dirname(model_path),
    )

    model_dir = final_model.save_dir
    return model_dir
