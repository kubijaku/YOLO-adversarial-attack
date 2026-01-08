# script generate confusion matrix's to enable easy comparison of the results for the model
# for validation data with new created adversarial images
import os
import warnings
from typing import Any

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---------------- util functions ----------------
def get_yolo_boxes(label_file_path: str) -> list[tuple[int, float, float, float, float]]:
    """
    Parse a YOLO-format label file and return bounding boxes.

    Each line in the label file should have the format:
        <class> <x_center> <y_center> <width> <height>

    Args:
        label_file_path (str): Path to the YOLO label file.

    Returns:
        list[tuple[int, float, float, float, float]]:
            List of tuples representing each bounding box as
            (class_id, x_center, y_center, width, height).
            Returns an empty list if the file does not exist or is empty.
    """
    boxes: list[tuple[int, float, float, float, float]] = []
    if not os.path.exists(label_file_path):
        return boxes

    with open(label_file_path, "r") as label_file:
        for line in label_file:
            line_parts = line.strip().split()
            if len(line_parts) >= 5:
                class_id = int(float(line_parts[0]))
                x_center, y_center, width, height = map(float, line_parts[1:5])
                boxes.append((class_id, x_center, y_center, width, height))

    return boxes


def yolo_norm_to_xyxy(
    x_center_norm: float,
    y_center_norm: float,
    width_norm: float,
    height_norm: float,
    image_width: int,
    image_height: int
) -> list[float]:
    """
    Convert YOLO normalized bounding box coordinates to absolute (x1, y1, x2, y2) format.

    YOLO format:
        - x_center_norm, y_center_norm: normalized center coordinates (0-1)
        - width_norm, height_norm: normalized width and height (0-1)

    Args:
        x_center_norm (float): Normalized x center of the bounding box.
        y_center_norm (float): Normalized y center of the bounding box.
        width_norm (float): Normalized width of the bounding box.
        height_norm (float): Normalized height of the bounding box.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.

    Returns:
        list[float]: [x1, y1, x2, y2] coordinates in pixel space.
    """
    x_center_abs = x_center_norm * image_width
    y_center_abs = y_center_norm * image_height
    box_width_abs = width_norm * image_width
    box_height_abs = height_norm * image_height
    x1 = x_center_abs - box_width_abs / 2
    y1 = y_center_abs - box_height_abs / 2
    x2 = x_center_abs + box_width_abs / 2
    y2 = y_center_abs + box_height_abs / 2
    return [x1, y1, x2, y2]



def iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    """
    Compute the Intersection over Union (IoU) of two bounding boxes in (x1, y1, x2, y2) format.

    Args:
        box_a (list[float]): Bounding box A in format [x1, y1, x2, y2].
        box_b (list[float]): Bounding box B in format [x1, y1, x2, y2].

    Returns:
        float: Intersection over Union (IoU) value between 0.0 and 1.0.
               Returns 0.0 if boxes do not overlap.
    """
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    intersection_width = max(0.0, x_right - x_left)
    intersection_height = max(0.0, y_bottom - y_top)
    intersection_area = intersection_width * intersection_height

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def find_images_recursive(root_dir: str) -> list[str]:
    """
    Recursively find all image files under the given directory that match known image suffixes.

    Args:
        root_dir (str): Root directory to search for image files.

    Returns:
        list[str]: Sorted list of image file paths found under the directory.
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        return []

    image_files = [
        str(file_path)
        for file_path in root_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUFFIXES
    ]

    return sorted(image_files)

def load_images(folder_path: str) -> list[str]:
    """
    Load all image file paths from a folder, filtering by common image file extensions.

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        list[str]: Sorted list of image file paths with extensions
                   in SUFFIXES.
    """
    folder = Path(folder_path)
    image_files = sorted(
        [
            str(file_path)
            for file_path in folder.iterdir()
            if file_path.suffix.lower() in SUFFIXES
        ]
    )
    return image_files


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


def compute_iou_matrix(gt_objects: list[Any], predictions: list) -> np.ndarray:
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
    ground_truth_objects: list[Any],
    predicted_objects: list[Any],
    iou_matrix: np.ndarray,
    iou_threshold: float,
    confusion_matrix: np.ndarray,
) -> tuple[set[Any], set[Any]]:
    """
    Match ground truth objects with predicted objects based on IoU and update the confusion matrix.

    Args:
        ground_truth_objects (list[Any]): List of ground truth objects, each containing at least a 'cls' and 'xyxy'.
        predicted_objects (list[Any]): List of predicted objects, each containing at least a 'cls' and 'xyxy'.
        iou_matrix (np.ndarray): Precomputed IoU values between all ground truth and predicted boxes.
        iou_threshold (float): Minimum IoU to consider a match.
        confusion_matrix (np.ndarray): Confusion matrix to be updated with matches.

    Returns:
        tuple[set[Any], set[Any]]:
            - Set of indices of matched ground truth objects.
            - Set of indices of matched predicted objects.
    """
    matched_ground_truth_indices = set()
    matched_prediction_indices = set()

    while True:
        if iou_matrix.size == 0:
            break
        best_gt_idx, best_pred_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        max_iou = iou_matrix[best_gt_idx, best_pred_idx]
        if max_iou < iou_threshold:
            break
        ground_truth_class = ground_truth_objects[best_gt_idx]["cls"]
        predicted_class = predicted_objects[best_pred_idx]["cls"]
        confusion_matrix[ground_truth_class, predicted_class] += 1
        matched_ground_truth_indices.add(best_gt_idx)
        matched_prediction_indices.add(best_pred_idx)
        iou_matrix[best_gt_idx, :] = -1.0
        iou_matrix[:, best_pred_idx] = -1.0

    return matched_ground_truth_indices, matched_prediction_indices


# ---------------- core: evaluate one folder -> return local confusion_matrix ----------------
def get_gt_objects(label_file: str, image_path: str, image_width: int, image_height: int) -> list[Any]:
    """
    Load ground truth objects from a YOLO label file and convert normalized coordinates to absolute image coordinates.

    Args:
        label_file (str): Path to the YOLO label file.
        image_path (str): Path to the image file (used for warnings).
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.

    Returns:
        list[Any]: List of ground truth objects, each represented as a dictionary with keys:
            - "cls": integer class ID
            - "xyxy": list of absolute bounding box coordinates [x1, y1, x2, y2]

    Raises:
        LookupError: If no label file is provided.
    """
    ground_truth_objects = []

    if label_file:
        yolo_boxes = get_yolo_boxes(label_file)
        for class_id, x_center, y_center, width, height in yolo_boxes:
            ground_truth_objects.append(
                {
                    "cls": int(class_id),
                    "xyxy": yolo_norm_to_xyxy(x_center, y_center, width, height, image_width, image_height),
                }
            )
    else:
        print("Warning: no ground truth objects label found for", image_path, "- skipping")
        raise LookupError("No ground truth objects label found for the given image")

    return ground_truth_objects



def get_predictions(model_results: list, confidence_threshold: float, image_path: str) -> list[Any]:
    """
    Extract predictions from YOLO model results, filtering out low-confidence detections.

    Args:
        model_results (list): List of YOLO prediction results (usually the output of model.predict()).
        confidence_threshold (float): Minimum confidence required to include a prediction.
        image_path (str): Path to the image (used for warnings).

    Returns:
        list[Any]: List of predicted objects, each represented as a dictionary with keys:
            - "cls": integer class ID
            - "xyxy": list of bounding box coordinates [x1, y1, x2, y2]
            - "conf": confidence score (float)

    Warnings:
        - Warns if a prediction is below the confidence threshold.
        - Warns if no predictions are found for the image.
    """
    first_result = model_results[0]

    predictions = []

    boxes_xyxy = first_result.boxes.xyxy.cpu().numpy()
    confidences = first_result.boxes.conf.cpu().numpy()
    detected_class_ids = first_result.boxes.cls.cpu().numpy().astype(int)

    for box_coords, class_id, confidence in zip(boxes_xyxy, detected_class_ids, confidences):
        if confidence < confidence_threshold:
            warnings.warn(
                f"Prediction under confidence threshold {confidence_threshold} for {image_path} - skipping.",
                category=UserWarning,
                stacklevel=2,
            )
            continue
        predictions.append(
            {
                "cls": int(class_id),
                "xyxy": [float(box_coords[0]), float(box_coords[1]), float(box_coords[2]), float(box_coords[3])],
                "conf": float(confidence),
            }
        )

    if len(predictions) == 0:
        warnings.warn(
            f"No predictions found for {image_path}",
            category=UserWarning,
            stacklevel=2,
        )

    return predictions


def evaluate_confusion_matrix(
    images_dir: str,
    labels_dir: str,
    class_names: list,
    model_path: str,
    device: str,
    conf_threshold: float,
    iou_threshold: float,
) -> np.ndarray:
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
            label_file=label_file, image_path=img_path, image_width=img_w, image_height=img_h
        )

        results = model.predict(
            source=img_path, imgsz=max(img_w, img_h), conf=conf_threshold, device=device
        )

        predictions = get_predictions(
            model_results=results, confidence_threshold=conf_threshold, image_path=img_path
        )
        iou_matrix = compute_iou_matrix(gt_objects, predictions)

        matched_ground_truth, matched_pred = get_matched_ground_truth_and_predictions(
            ground_truth_objects=gt_objects,
            predicted_objects=predictions,
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
def normalize_confusion_matrix(confusion_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize a confusion matrix row-wise so that each row sums to 1.

    Args:
        confusion_matrix (np.ndarray): A 2D array representing the confusion matrix
            (shape: [num_classes, num_classes]).

    Returns:
        np.ndarray: Row-normalized confusion matrix with NaNs replaced by 0.0.
    """
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = confusion_matrix.astype(float) / row_sums
    normalized_matrix[np.isnan(normalized_matrix)] = 0.0
    return normalized_matrix



def plot_and_save_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    output_path: str = "confusion_matrix.png",
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot and save a confusion matrix as an image.

    Args:
        confusion_matrix (np.ndarray): 2D array representing the confusion matrix.
        class_names (list[str]): List of class names corresponding to the confusion matrix rows/columns.
        output_path (str, optional): File path to save the plotted image. Defaults to "confusion_matrix.png".
        title (str, optional): Title for the plot. Defaults to "Confusion Matrix".
    """
    labels = class_names + ["No Detection"]
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(confusion_matrix, interpolation="nearest", cmap="Blues")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)

    is_floating_type = np.issubdtype(confusion_matrix.dtype, np.floating)

    for row_idx in range(confusion_matrix.shape[0]):
        for col_idx in range(confusion_matrix.shape[1]):
            cell_value = confusion_matrix[row_idx, col_idx]
            text_label = f"{cell_value:.2f}" if is_floating_type else f"{int(cell_value)}"
            ax.text(
                col_idx,
                row_idx,
                text_label,
                ha="center",
                va="center",
                color="white" if cell_value > confusion_matrix.max() / 2 else "black",
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print("Saved confusion matrix image to:", output_path)


def save_confusion_matrix_as_csv(
    confusion_matrix: np.ndarray,
    csv_filename: str,
    output_directory: str
) -> None:
    """
    Save a confusion matrix as a CSV file.

    Args:
        confusion_matrix (np.ndarray): 2D array representing the confusion matrix.
        csv_filename (str): Name of the CSV file to save.
        output_directory (str): Directory path where the CSV file will be saved.
    """
    csv_file_path = os.path.join(output_directory, csv_filename)
    np.savetxt(
        csv_file_path,
        confusion_matrix,
        fmt="%d",
        delimiter=",",
    )
    print("Saved CSV to", csv_file_path)


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
