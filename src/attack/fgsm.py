import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images_recursive(root):
    p = Path(root)
    if not p.exists():
        return []
    files = [str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUFFIXES]
    return sorted(files)


def load_image_tensor(path, device):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
    return t, arr


def save_tensor_image(tensor, path):
    t = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    im = Image.fromarray((t * 255).astype("uint8"))
    im.save(path)


def read_yolo_label_file(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(float(parts[0]));
                xc, yc, w, h = map(float, parts[1:5])
                boxes.append((cls, xc, yc, w, h))
    return boxes


def flatten_pred_tensor(p: torch.Tensor):
    """
    Convert a raw pred tensor (B, C, H, W) to shape (B, P, C) where:
     - B is number of batches,
     - P is number of predicted boxes/proposals,
     - C is number channel size,
     - H and W: the grid sizes for different detection scales
    Accepts tensors shaped like:
      - (B, C, H, W)
    """
    if not isinstance(p, torch.Tensor):
        raise TypeError("Expected torch.Tensor")
    # For dims >=4, try to bring channels to last dim
    # Our case: (B, C, H, W)
    t = p
    if t.dim() >= 4:
        # heuristics: if dim 1 is small (<32) and last dims >1, it might be channel
        # We try to transpose so last dim is channels
        # Case: (B, C, H, W) -> (B, H, W, C)
        if t.shape[1] <= 1024 and t.shape[-1] <= 1024:
            # try common (B,C,H,W)
            try:
                t = t.permute(0, 2, 3, 1)  # (B,H,W,C)
            except Exception:
                pass
        # Flatten spatial dims into P
        B = t.shape[0]
        rest = t.shape[1:]
        P = 1
        for d in rest[:-1]:
            P *= d
        C = rest[-1]
        t = t.reshape(B, P, C)
        return t

    # fallback: flatten everything except batch
    B = p.shape[0]
    flat = p.reshape(B, -1, p.shape[-1])

    return flat


def compute_proxy_from_preds(raw, gt_boxes=None):
    """
    raw: either torch.Tensor or list/tuple of tensors (various shapes).
    in our case it is a list of 3 tensors (each one for diffrent spatial resolutions:
        - small objects (high-resolution feature map),
        - medium objects,
        - large objects (low-resolution feature map).
    Compute a proxy loss robustly without concatenating incompatible tensors.
    """
    # normalize to list of tensors
    if isinstance(raw, torch.Tensor):
        pred_list = [raw]
    elif isinstance(raw, (list, tuple)):
        pred_list = [p for p in raw if isinstance(p, torch.Tensor)]
    else:
        raise RuntimeError("Unsupported preds type")

    device = pred_list[0].device if pred_list else torch.device(DEVICE)
    total_loss = torch.tensor(0.0, device=device)

    per_head_scores = []
    for p in pred_list:
        try:
            flat = flatten_pred_tensor(p)  # (B,P,C)
        except Exception:
            # fallback: convert to float and sum
            total_loss = total_loss + p.float().abs().sum()
            continue

        B, P, C = flat.shape
        if C < 5:
            print('never comes in here')
            # treat as generic activation summary
            per_head_scores.append(flat.abs().sum(dim=(1, 2)))  # (B,)
            continue

        obj = flat[..., 4]  # (B,P)

        # If we have class scores → use channels 5+ (standard YOLO)
        # If not → fallback and treat object as a "fake" class logit (for safety)
        class_logits = flat[..., 5:] if C > 5 else flat[..., 4:].unsqueeze(-1)
        if class_logits.shape[-1] == 1:
            class_probs = torch.sigmoid(class_logits)
        else:
            # softmax across class dim
            class_probs = F.softmax(class_logits, dim=-1)

        max_class_prob, _ = class_probs.max(dim=-1)  # (B,P)
        score = torch.sigmoid(obj) * max_class_prob  # (B,P)
        per_head_scores.append(score)  # keep per-pred scores
    # Now aggregate per-head scores into a single loss
    # If ground truth boxes available, penalize the model's highest score for those classes.
    if gt_boxes:
        # For each GT class, find its best score across all heads & preds
        for (cls, xc, yc, w, h) in gt_boxes:
            cls_idx = int(cls)
            best_vals = []
            for head_score in per_head_scores:
                # head_score is (B,P); class-specific proxy: need class_probs for that head if available
                # But we simplified: if head provided only score, we use that score as proxy.
                # For heads where we computed class_probs we would have used class-specific; here we approximate.
                # We'll use the head_score's max as a conservative proxy.
                best_vals.append(head_score.max(dim=1).values)  # (B,)
            if not best_vals:
                continue
            # stack and take the maximum across heads, then -log to make loss we maximize
            stacked = torch.stack(best_vals, dim=0)  # (num_heads, B)
            max_across = stacked.max(dim=0).values  # (B,)
            total_loss = total_loss + (-torch.log(max_across + 1e-6)).sum()
    else:
        # no GT: just sum all per-head scores (we will maximize this)
        for head_score in per_head_scores:
            total_loss = total_loss + head_score.sum()

    return total_loss


def fgsm_attack(model: str, img_dir: str, labels_dir: str, out: str, pert_with_eps_dir: str, pert_dir: str, eps: float,
                max_img: int, device: str):
    global MODEL_PATH
    global VAL_IMAGES_DIR
    global VAL_LABELS_DIR
    global OUTPUT_DIR
    global PER_IMG_DIR_WITH_EPS
    global PER_IMG_DIR
    global EPSILON
    global MAX_IMAGES
    global DEVICE

    MODEL_PATH = model
    VAL_IMAGES_DIR = img_dir
    VAL_LABELS_DIR = labels_dir
    OUTPUT_DIR = out
    PER_IMG_DIR_WITH_EPS = pert_with_eps_dir
    PER_IMG_DIR = pert_dir
    EPSILON = eps
    MAX_IMAGES = max_img
    DEVICE = device

    # print current dir
    print("Device:", DEVICE)
    print("Loading model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    internal = getattr(model, "model", None)
    if internal is None:
        print("Warning: model.model not found; attempting wrapper forward which may be non-differentiable.")
    else:
        print("Using internal model:", type(internal))

    imgs = find_images_recursive(VAL_IMAGES_DIR)
    print(f"Found {len(imgs)} images under {VAL_IMAGES_DIR}")
    if len(imgs) > 0:
        print("First few images:")
        for p in imgs[:10]:
            print("  ", p)
    if not imgs:
        print("No images found. Exiting.")
        return

    if MAX_IMAGES:
        imgs = imgs[:MAX_IMAGES]

    for i, img_path in enumerate(imgs, 1):
        print(f"\n[{i}/{len(imgs)}] {img_path}")
        base = Path(img_path).stem
        label_path = os.path.join(VAL_LABELS_DIR, base + ".txt")
        gt_boxes = read_yolo_label_file(label_path)
        if gt_boxes:
            print(f" - Found {len(gt_boxes)} GT boxes")
        else:
            print(" - No GT boxes (will use fallback loss)")

        img_t, orig = load_image_tensor(img_path, DEVICE)
        img_t = img_t.clone().detach()
        img_t.requires_grad = True

        # Get raw preds via internal model (differentiable)
        try:
            internal_model = getattr(model, "model", model)
            internal_model.train()
            raw = internal_model(img_t)  # raw can be tensor or list/tuple of tensors
        except Exception as e:
            print(" - Internal forward failed:", e)
            try:
                # fallback to wrapper call (non-diff) -> skip
                _ = model(img_path)
                print(" - Wrapper call returned non-differentiable results. Skipping image.")
                continue
            except Exception:
                print(" - Wrapper call also failed. Skipping image.")
                continue

        # Build proxy loss and do FGSM
        try:
            loss = compute_proxy_from_preds(raw, gt_boxes)
            # maximize loss -> gradient step in direction of sign(grad)
            loss.backward()
            grad = img_t.grad.data
            if grad is None:
                print(" - No gradient computed (None). Skipping.")
                continue
            perturbations = torch.sign(grad)

            out_path = os.path.join(PER_IMG_DIR, f"{base}_eps{int(EPSILON * 255)}.png")
            save_tensor_image(perturbations, out_path)

            out_path = os.path.join(PER_IMG_DIR_WITH_EPS, f"{base}_eps{int(EPSILON * 255)}.png")
            save_tensor_image(EPSILON * perturbations, out_path)

            # adding perturbations
            adv = img_t + EPSILON * perturbations
            # normalizing the image
            adv = torch.clamp(adv, 0.0, 1.0).detach()

            out_path = os.path.join(OUTPUT_DIR, f"{base}_eps{int(EPSILON * 255)}.png")
            save_tensor_image(adv, out_path)

            print(
                f" - Saved adversarial to {out_path} (loss {float(loss):.4f}, grad_max {float(grad.abs().max()):.6f})")
        except Exception as e:
            print(" - Attack failed for this image:", e)
            continue

    print("\nDone. Check folder:", OUTPUT_DIR)
