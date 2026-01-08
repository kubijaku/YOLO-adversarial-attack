import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

from utils.utils import get_yolo_boxes, find_images_recursive


def load_image_tensor(image_path: str, device: str) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Loads an image from disk and converts it to both a PyTorch tensor and a NumPy array.

    Args:
        image_path (str): Path to the image file.
        device (str): PyTorch device to move the tensor to (e.g., 'cpu', 'cuda', 'mps').

    Returns:
        Tuple[torch.Tensor, np.ndarray]:
            - tensor: Image tensor of shape (1, 3, H, W) with values in [0, 1].
            - array: NumPy array of shape (H, W, 3) with values in [0, 1].
    """
    pil_image = Image.open(image_path).convert("RGB")
    image_array = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
    return image_tensor, image_array


def save_tensor_image(tensor: torch.Tensor, output_path: str) -> None:
    """
    Saves a PyTorch tensor as an image file.

    Args:
        tensor (torch.Tensor): Image tensor of shape (1, 3, H, W) or similar.
        output_path (str): Path to save the resulting image.

    Notes:
        - Tensor values are expected in the range [0, 1].
        - Converts tensor to CPU, clamps values, converts to uint8, and saves as image.
    """
    image_array = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    image = Image.fromarray((image_array * 255).astype("uint8"))
    image.save(output_path)


def read_yolo_label_file(label_file_path) -> list[tuple[int, float, float, float, float]]:
    """
    Reads a YOLO-format label file and returns a list of bounding boxes.

    Each line in the file is expected to contain:
        <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>

    Returns:
        List of tuples: (class_id, x_center, y_center, width, height)
    """
    yolo_boxes: list[tuple[int, float, float, float, float]] = []

    if not os.path.exists(label_file_path):
        return yolo_boxes

    with open(label_file_path, "r") as label_file:
        for line in label_file:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(float(parts[0]))
                x_center, y_center, width, height = map(float, parts[1:5])
                yolo_boxes.append((class_id, x_center, y_center, width, height))

    return yolo_boxes



def flatten_pred_tensor(prediction_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a raw pred tensor (B, C, H, W) to shape (B, P, C) where:
     - B is number of batches,
     - P is number of predicted boxes/proposals,
     - C is number channel size,
     - H and W: the grid sizes for different detection scales
    Accepts tensors shaped like:
      - (B, C, H, W)
    """
    if not isinstance(prediction_tensor, torch.Tensor):
        raise TypeError("Expected torch.Tensor")

    if prediction_tensor.dim() >= 4:
        # heuristics: if dim 1 is small (<1024) and last dims >1, it might be channel
        # Case: (B, C, H, W) -> (B, H, W, C)
        if prediction_tensor.shape[1] <= 1024 and prediction_tensor.shape[-1] <= 1024:
            prediction_tensor = prediction_tensor.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Flatten spatial dimensions into prediction dimension
        batch_size = prediction_tensor.shape[0]
        remaining_dimensions = prediction_tensor.shape[1:]

        num_predictions = 1
        for spatial_dim in remaining_dimensions[:-1]:
            num_predictions *= spatial_dim

        channel_size = remaining_dimensions[-1]

        reshaped_tensor = prediction_tensor.reshape(
            batch_size, num_predictions, channel_size
        )
        return reshaped_tensor

    # fallback: flatten everything except batch dimension
    batch_size = prediction_tensor.shape[0]
    flattened_tensor = prediction_tensor.reshape(
        batch_size, -1, prediction_tensor.shape[-1]
    )

    return flattened_tensor


def compute_proxy_from_preds(
    raw_predictions,
    device: str | torch._C.device,
    gt_boxes=None,
) -> torch.Tensor:
    """
    raw_predictions: either torch.Tensor or list/tuple of tensors (various shapes).
    In our case it is a list of 3 tensors (each one for different spatial resolutions:
        - small objects (high-resolution feature map),
        - medium objects,
        - large objects (low-resolution feature map).
    Compute a proxy loss robustly without concatenating incompatible tensors.
    """
    # normalize to list of tensors
    if isinstance(raw_predictions, torch.Tensor):
        prediction_tensors = [raw_predictions]
    elif isinstance(raw_predictions, (list, tuple)):
        prediction_tensors = [
            prediction_tensor
            for prediction_tensor in raw_predictions
            if isinstance(prediction_tensor, torch.Tensor)
        ]
    else:
        raise RuntimeError("Unsupported preds type")

    torch_device = (
        prediction_tensors[0].device
        if prediction_tensors
        else torch.device(device)
    )
    total_loss = torch.tensor(0.0, device=torch_device)

    per_head_scores = []
    for prediction_tensor in prediction_tensors:
        try:
            flattened_predictions = flatten_pred_tensor(
                prediction_tensor
            )  # (B, P, C)
        except Exception:
            # fallback: convert to float and sum
            total_loss = total_loss + prediction_tensor.float().abs().sum()
            continue

        batch_size, num_predictions, channel_size = flattened_predictions.shape

        objectness_logits = flattened_predictions[..., 4]  # (B, P)

        # If we have class scores → use channels 5+ (standard YOLO)
        # If not → fallback and treat object as a "fake" class logit (for safety)
        class_logits = (
            flattened_predictions[..., 5:]
            if channel_size > 5
            else flattened_predictions[..., 4:].unsqueeze(-1)
        )

        if class_logits.shape[-1] == 1:
            class_probabilities = torch.sigmoid(class_logits)
        else:
            # softmax across class dim
            class_probabilities = F.softmax(class_logits, dim=-1)

        max_class_probability, _ = class_probabilities.max(dim=-1)  # (B, P)
        detection_scores = (
            torch.sigmoid(objectness_logits) * max_class_probability
        )  # (B, P)

        per_head_scores.append(detection_scores)

    # Now aggregate per-head scores into a single loss
    # If ground truth boxes available, penalize the model's highest score for those classes.
    if gt_boxes:
        # For each GT class, find its best score across all heads & preds
        for class_id, center_x, center_y, width, height in gt_boxes:
            best_scores_per_head = []
            for head_scores in per_head_scores:
                best_scores_per_head.append(
                    head_scores.max(dim=1).values
                )  # (B,)

            if not best_scores_per_head:
                continue

            stacked_scores = torch.stack(best_scores_per_head, dim=0)  # (num_heads, B)
            max_scores_across_heads = stacked_scores.max(dim=0).values  # (B,)

            total_loss = total_loss + (
                -torch.log(max_scores_across_heads + 1e-6)
            ).sum()
    else:
        # no GT: just sum all per-head scores (we will maximize this)
        for head_scores in per_head_scores:
            total_loss = total_loss + head_scores.sum()

    return total_loss


def fgsm_attack(
    model_path: str,
    img_dir: str,
    labels_dir: str,
    adv_img_dir: str,
    pert_with_eps_dir: str,
    pert_dir: str,
    eps: float,
    max_img: int,
    device: str,
):
    """
    Perform an untargeted FGSM (Fast Gradient Sign Method) adversarial attack
    against a YOLO model on a directory of images.

    For each image:
    - Loads the image and corresponding YOLO labels (if available)
    - Computes a proxy loss from raw model predictions
    - Generates adversarial perturbations using FGSM
    - Saves:
        * raw perturbations
        * perturbations scaled by epsilon
        * final adversarial images

    Args:
        model_path (str): Path to the YOLO model (.pt file).
        img_dir (str): Directory containing input images.
        labels_dir (str): Directory containing YOLO-format label files.
        adv_img_dir (str): Directory to save adversarial images.
        pert_with_eps_dir (str): Directory to save epsilon-scaled perturbations.
        pert_dir (str): Directory to save raw perturbation images.
        eps (float): FGSM epsilon (perturbation strength).
        max_img (int): Maximum number of images to process (0 = all).
        device (str): Device identifier ("cpu", "cuda", or "mps").
    """
    print("Device:", device)
    print("Loading model:", model_path)

    yolo_model = YOLO(model_path)
    yolo_model.to(device)

    internal_model_attr = getattr(yolo_model, "model", None)
    if internal_model_attr is None:
        print(
            "Warning: model.model not found; attempting wrapper forward which may be non-differentiable."
        )
    else:
        print("Using internal model:", type(internal_model_attr))

    image_paths = find_images_recursive(img_dir)
    print(f"Found {len(image_paths)} images under {img_dir}")

    if len(image_paths) > 0:
        print("First few images:")
        for image_path in image_paths[:10]:
            print("  ", image_path)

    if not image_paths:
        print("No images found. Exiting.")
        return

    if max_img:
        image_paths = image_paths[:max_img]

    for image_index, image_path in enumerate(image_paths, 1):
        print(f"\n[{image_index}/{len(image_paths)}] {image_path}")

        image_stem = Path(image_path).stem
        label_file_path = os.path.join(labels_dir, image_stem + ".txt")

        ground_truth_boxes = get_yolo_boxes(label_file_path)
        if ground_truth_boxes:
            print(f" - Found {len(ground_truth_boxes)} GT boxes")
        else:
            print(" - No GT boxes (will use fallback loss)")

        image_tensor, original_image_array = load_image_tensor(
            image_path, device=device
        )
        image_tensor = image_tensor.clone().detach()
        image_tensor.requires_grad = True

        # Get raw predictions via internal model (differentiable)
        internal_model = getattr(yolo_model, "model", yolo_model)
        internal_model.train()

        raw_predictions = internal_model(image_tensor)

        # Build proxy loss and apply FGSM
        try:
            loss = compute_proxy_from_preds(
                raw_predictions=raw_predictions,
                gt_boxes=ground_truth_boxes,
                device=device,
            )

            # maximize loss -> gradient step in direction of sign(grad)
            loss.backward()

            gradient_tensor = image_tensor.grad.data
            if gradient_tensor is None:
                print(" - No gradient computed (None). Skipping.")
                continue

            perturbation_tensor = torch.sign(gradient_tensor)

            perturbation_path = os.path.join(
                pert_dir, f"{image_stem}_eps{int(eps * 255)}.png"
            )
            save_tensor_image(perturbation_tensor, perturbation_path)

            scaled_perturbation_path = os.path.join(
                pert_with_eps_dir, f"{image_stem}_eps{int(eps * 255)}.png"
            )
            save_tensor_image(eps * perturbation_tensor, scaled_perturbation_path)

            # apply perturbation
            adversarial_tensor = image_tensor + eps * perturbation_tensor
            adversarial_tensor = torch.clamp(adversarial_tensor, 0.0, 1.0).detach()

            adversarial_output_path = os.path.join(
                adv_img_dir, f"{image_stem}_eps{int(eps * 255)}.png"
            )
            save_tensor_image(adversarial_tensor, adversarial_output_path)

            print(
                f" - Saved adversarial to {adversarial_output_path} "
                f"(loss {float(loss):.4f}, grad_max {float(gradient_tensor.abs().max()):.6f})"
            )

        except Exception as exception:
            print(" - Attack failed for this image:", exception)
            continue

    print("\nDone. Check folder:", adversarial_output_path)
