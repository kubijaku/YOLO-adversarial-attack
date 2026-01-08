import torch
import argparse
import os

from attack.fgsm import fgsm_attack

from utils.utils import (
    evaluate_confusion_matrix,
    normalize_confusion_matrix,
    plot_and_save_confusion_matrix,
    train_model,
    save_confusion_matrix_as_csv,
)

# ---------------- USER CONFIG ----------------
CLASS_NAMES = [
    "Green Light",
    "Red Light",
    "Speed Limit 10",
    "Speed Limit 100",
    "Speed Limit 110",
    "Speed Limit 120",
    "Speed Limit 20",
    "Speed Limit 30",
    "Speed Limit 40",
    "Speed Limit 50",
    "Speed Limit 60",
    "Speed Limit 70",
    "Speed Limit 80",
    "Speed Limit 90",
    "Stop",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adversarial attack generator for YOLO models"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="./model/train/weights/best.pt",
        help="Path to YOLO model .pt file",
    )

    parser.add_argument(
        "--img-dir",
        type=str,
        default="./car/valid/images",
        help="Validation image directory",
    )

    parser.add_argument(
        "--labels-dir",
        type=str,
        default="./car/valid/labels",
        help="Validation label directory",
    )

    parser.add_argument(
        "--adv-img-dir",
        type=str,
        default="./results/adv_images",
        help="Directory to save adversarial images",
    )

    parser.add_argument(
        "--confusion-matrix-dir",
        type=str,
        default="./results/attack_results",
        help="Directory to save confusion matrix's and connected .csv files",
    )

    parser.add_argument(
        "--pert-with-eps-dir",
        type=str,
        default="./results/pert_img_with_eps",
        help="Directory to save perturbation*epsilon visualizations",
    )

    parser.add_argument(
        "--pert-dir",
        type=str,
        default="./results/pert_img",
        help="Directory to save raw perturbation images",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=8.0 / 255.0,
        help="Epsilon (FGSM strength), default=8/255",
    )

    parser.add_argument(
        "--max-img",
        type=int,
        default=0,
        help="Process only the first N images, 0 = all",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        choices=["cpu", "cuda", "mps"],
        help="Which device to use",
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions to labels",
    )

    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.001,
        help="Confidence threshold: filter predictions below this value",
    )

    parser.add_argument("--train", type=bool, default=False, help="Train the model")

    return parser.parse_args()


def print_configuration(args: argparse.Namespace) -> None:
    """
    param: args: argparse.Namespace - arguments parsed by argparse
    function prints arguments
    """
    print("\n========== CONFIGURATION ==========")
    print(f"Model path:              {args.model}")
    print(f"Validation images:       {args.img_dir}")
    print(f"Validation labels:       {args.labels_dir}")
    print(f"Save adversarial to:     {args.adv_img_dir}")
    print(f"Save pert*eps to:        {args.pert_with_eps_dir}")
    print(f"Save raw perturbations:  {args.pert_dir}")
    print(f"Epsilon:                 {args.eps}")
    print(f"Max images:              {args.max_img}")
    print(f"Device:                  {args.device}")
    print(f"IoU threshold :          {args.iou_threshold}")
    print(f"Conf-threshold:          {args.conf_threshold}")
    print(f"Train:                   {args.train}")
    print("===================================\n")


def create_directories(dir_list: list) -> None:
    """
    param: dir_list: list of directory paths
    Function creates needed directories
    """
    for dir_path in dir_list:
        os.makedirs(dir_path, exist_ok=True)


def main():
    args = parse_args()
    print_configuration(args)

    create_directories(
        [
            args.adv_img_dir,
            args.pert_with_eps_dir,
            args.pert_dir,
            args.confusion_matrix_dir,
        ]
    )

    if args.train:
        model_dir = train_model()
    else:
        model_dir = args.model

    fgsm_attack(
        model_path=model_dir,
        img_dir=args.img_dir,
        labels_dir=args.labels_dir,
        adv_img_dir=args.adv_img_dir,
        pert_with_eps_dir=args.pert_with_eps_dir,
        pert_dir=args.pert_dir,
        eps=args.eps,
        max_img=args.max_img,
        device=args.device,
    )

    ### Evaluate on validation (clean) set ###
    confusion_matrix_val = evaluate_confusion_matrix(
        args.img_dir,
        args.labels_dir,
        class_names=CLASS_NAMES,
        model_path=model_dir,
        device=args.device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
    )

    plot_and_save_confusion_matrix(
        confusion_matrix_val,
        CLASS_NAMES,
        out_path=os.path.join(
            args.confusion_matrix_dir, "val_dataset_confusion_matrix.png"
        ),
        title="Validation (clean) - Confusion Matrix (Normalized)",
    )

    confusion_matrix_val_norm = normalize_confusion_matrix(confusion_matrix_val)
    plot_and_save_confusion_matrix(
        confusion_matrix_val_norm,
        CLASS_NAMES,
        out_path=os.path.join(
            args.confusion_matrix_dir, "val_dataset_confusion_matrix_normalized.png"
        ),
        title="Validation (clean) - Confusion Matrix (Normalized)",
    )

    save_confusion_matrix_as_csv(
        confusion_matrix=confusion_matrix_val_norm,
        csv_filename="val_confusion_normalized.csv",
        output_dir=args.confusion_matrix_dir,
    )

    ### Evaluate on adversarial set ###
    confusion_matrix_adv = evaluate_confusion_matrix(
        args.adv_img_dir,
        args.labels_dir,
        class_names=CLASS_NAMES,
        model_path=model_dir,
        device=args.device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
    )

    plot_and_save_confusion_matrix(
        confusion_matrix_adv,
        CLASS_NAMES,
        out_path=os.path.join(
            args.confusion_matrix_dir, "adversarial_dataset_confusion_matrix.png"
        ),
        title="Validation (clean) - Confusion Matrix (Normalized)",
    )

    confusion_matrix_adv_norm = normalize_confusion_matrix(confusion_matrix_adv)
    plot_and_save_confusion_matrix(
        confusion_matrix_adv_norm,
        CLASS_NAMES,
        out_path=os.path.join(
            args.confusion_matrix_dir,
            "adversarial_dataset_confusion_matrix_normalized.png",
        ),
        title="Validation (clean) - Confusion Matrix (Normalized)",
    )

    save_confusion_matrix_as_csv(
        confusion_matrix=confusion_matrix_adv_norm,
        csv_filename="adversarial_confusion_normalized.csv",
        output_dir=args.confusion_matrix_dir,
    )


if __name__ == "__main__":
    main()
