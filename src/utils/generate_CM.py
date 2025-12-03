# script generate confusion matrix's to enable easy comparison of the results for the model
# for validation data with new created adversarial images
import os
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

# ---------------- USER CONFIG ----------------
CLASS_NAMES = [
    'Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110',
    'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40',
    'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80',
    'Speed Limit 90', 'Stop'
]

# ------------------------------------------------

NUM_CLASSES = len(CLASS_NAMES)
BACKGROUND = NUM_CLASSES  # index for background / no-detection


# ---------------- util functions ----------------
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
        [str(f) for f in p.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]])
    return imgs


# ---------------- core: evaluate one folder -> return local CM ----------------
def evaluate_CM(images_dir, labels_dir):
    """
    Evaluate detections on images in images_dir using GT labels in labels_dir.
    Returns a NEW confusion matrix of shape (NUM_CLASSES+1, NUM_CLASSES+1).
    """
    # create local confusion matrix (do not reuse global)
    cm = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=int)

    print("Loading model for inference:", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    try:
        model.to(DEVICE)
    except Exception:
        pass

    images = load_images(images_dir)
    print(f"Found {len(images)} images in {images_dir}")

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        # try to find corresponding GT label in labels_dir by matching basename prefix
        label_file = None
        for f in Path(labels_dir).rglob("*.txt"):
            if Path(f).stem in Path(img_path).stem:
                label_file = str(f)
                break
        if label_file is None:
            candidate = os.path.join(labels_dir, Path(img_path).stem + ".txt")
            if os.path.exists(candidate):
                label_file = candidate

        gt = []
        if label_file:
            yolo_boxes = read_yolo_label_file(label_file)
            for (cls, xc, yc, w, h) in yolo_boxes:
                gt.append({"cls": int(cls), "xyxy": yolo_norm_to_xyxy(xc, yc, w, h, img_w, img_h)})
        else:
            print("Warning: no GT label found for", img_path, "- skipping")
            continue

        results = model.predict(source=img_path, imgsz=max(img_w, img_h), conf=CONF_THR, device=DEVICE)
        r = results[0]
        preds = []
        try:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            for b, c, cf in zip(boxes_xyxy, classes, confs):
                if cf < CONF_THR:
                    continue
                preds.append(
                    {"cls": int(c), "xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])], "conf": float(cf)})
        except Exception:
            try:
                data = getattr(r.boxes, "data", None)
                if data is not None:
                    arr = data.cpu().numpy()
                    for row in arr:
                        x1, y1, x2, y2, cf, cl = row[:6]
                        if cf < CONF_THR:
                            continue
                        preds.append({"cls": int(cl), "xyxy": [x1, y1, x2, y2], "conf": float(cf)})
            except Exception:
                pass

        G = len(gt);
        P = len(preds)
        if P == 0:
            # all GTs are missed
            for gi in range(G):
                cm[gt[gi]["cls"], BACKGROUND] += 1
            continue

        iou_mat = np.zeros((G, P), dtype=float)
        for gi in range(G):
            for pj in range(P):
                iou_mat[gi, pj] = iou_xyxy(gt[gi]["xyxy"], preds[pj]["xyxy"])

        matched_gt = set()
        matched_pred = set()
        while True:
            if iou_mat.size == 0:
                break
            gi, pj = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            best_iou = iou_mat[gi, pj]
            if best_iou < IOU_THR:
                break
            gt_cls = gt[gi]["cls"]
            pred_cls = preds[pj]["cls"]
            cm[gt_cls, pred_cls] += 1
            matched_gt.add(gi)
            matched_pred.add(pj)
            iou_mat[gi, :] = -1.0
            iou_mat[:, pj] = -1.0

        for gi in range(G):
            if gi not in matched_gt:
                cm[gt[gi]["cls"], BACKGROUND] += 1

        for pj in range(P):
            if pj not in matched_pred:
                cm[BACKGROUND, preds[pj]["cls"]] += 1

    return cm


# ---------------- plotting / normalization ----------------
def normalize_confusion_matrix(cm):
    with np.errstate(all='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        normalized = cm.astype(float) / row_sums
        normalized[np.isnan(normalized)] = 0.0
    return normalized


def plot_confusion_matrix(cm, names, out_path="confusion_matrix.png", title="Confusion Matrix"):
    labels = names + ["No Detection"]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    is_float = np.issubdtype(cm.dtype, np.floating)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text = f"{value:.2f}" if is_float else f"{int(value)}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if value > cm.max() / 2 else "black")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print("Saved confusion matrix image to:", out_path)

    # generate_cm(args.img_out, args.img_dir, args.labels_dir, args.model, args.device, args.cm_out)


def generate_cm(adv_img_dir: str, img_dir: str, labels_dir: str, model: str, device: str, cm_out: str, iou_thr: str,
                conf_thr: str):
    global ADV_DIR
    global VAL_IMAGES_DIR
    global VAL_LABELS_DIR
    global MODEL_PATH
    global DEVICE
    global IOU_THR
    global CONF_THR
    global OUTPUT_DIR

    ADV_DIR = adv_img_dir
    VAL_IMAGES_DIR = img_dir
    VAL_LABELS_DIR = labels_dir
    MODEL_PATH = model
    DEVICE = device
    IOU_THR = iou_thr
    CONF_THR = conf_thr
    OUTPUT_DIR = cm_out

    # Evaluate on validation (clean) set
    cm_val = evaluate_CM(VAL_IMAGES_DIR, VAL_LABELS_DIR)
    cm_val_norm = normalize_confusion_matrix(cm_val)
    plot_confusion_matrix(cm_val_norm, CLASS_NAMES,
                          out_path=os.path.join(OUTPUT_DIR, "val_dataset_confusion_matrix_normalized.png"),
                          title="Validation (clean) - Confusion Matrix (Normalized)")

    # Evaluate on adversarial set
    cm_adv = evaluate_CM(ADV_DIR, VAL_LABELS_DIR)
    cm_adv_norm = normalize_confusion_matrix(cm_adv)
    plot_confusion_matrix(cm_adv_norm, CLASS_NAMES,
                          out_path=os.path.join(OUTPUT_DIR, "after_attack_confusion_matrix_normalized.png"),
                          title="After Attack - Confusion Matrix (Normalized)")

    # Save raw CSVs as well
    np.savetxt(os.path.join(OUTPUT_DIR, "val_confusion_raw.csv"), cm_val, fmt="%d", delimiter=",")
    np.savetxt(os.path.join(OUTPUT_DIR, "adv_confusion_raw.csv"), cm_adv, fmt="%d", delimiter=",")
    np.savetxt(os.path.join(OUTPUT_DIR, "val_confusion_norm.csv"), cm_val_norm, fmt="%.4f", delimiter=",")
    np.savetxt(os.path.join(OUTPUT_DIR, "adv_confusion_norm.csv"), cm_adv_norm, fmt="%.4f", delimiter=",")

    print("Saved CSVs and PNGs to", OUTPUT_DIR)
