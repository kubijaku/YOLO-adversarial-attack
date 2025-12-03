import torch
import argparse
import os

from attack.fgsm import fgsm_attack
from utils.generate_CM import generate_cm
from utils.train_model import train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adversarial attack generator for YOLO models"
    )

    parser.add_argument("--model", type=str, default="../model/train/weights/best.pt",
                        help="Path to YOLO model .pt file")

    parser.add_argument("--img-dir", type=str, default="./car/valid/images",
                        help="Validation image directory")

    parser.add_argument("--labels-dir", type=str, default="./car/valid/labels",
                        help="Validation label directory")

    parser.add_argument("--img_out", type=str, default="../results/adv_images",
                        help="Directory to save adversarial images")

    parser.add_argument("--cm_out", type=str, default="../results/attack_results",
                        help="Directory to save confusion matrix's and connected .csv files")

    parser.add_argument("--pert-with-eps-dir", type=str, default="../results/pert_img_with_eps",
                        help="Directory to save perturbation*epsilon visualizations")

    parser.add_argument("--pert-dir", type=str, default="../results/pert_img",
                        help="Directory to save raw perturbation images")

    parser.add_argument("--eps", type=float, default=8.0 / 255.0,
                        help="Epsilon (FGSM strength), default=8/255")

    parser.add_argument("--max-img", type=int, default=0,
                        help="Process only the first N images, 0 = all")

    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Which device to use")

    parser.add_argument("--iou-thr", type=float, default=0.5,
                        help="IoU threshold for matching predictions to labels")

    parser.add_argument("--conf-thr", type=float, default=0.001,
                        help="Confidence threshold: filter predictions below this value")

    parser.add_argument("--train", type=bool, default=False,
                        help="Train the model")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create needed directories
    os.makedirs(args.img_out, exist_ok=True)
    os.makedirs(args.pert_with_eps_dir, exist_ok=True)
    os.makedirs(args.pert_dir, exist_ok=True)
    os.makedirs(args.cm_out, exist_ok=True)

    print("\n========== CONFIGURATION ==========")
    print(f"Model path:              {args.model}")
    print(f"Validation images:       {args.img_dir}")
    print(f"Validation labels:       {args.labels_dir}")
    print(f"Save adversarial to:     {args.img_out}")
    print(f"Save pert*eps to:        {args.pert_with_eps_dir}")
    print(f"Save raw perturbations:  {args.pert_dir}")
    print(f"Epsilon:                 {args.eps}")
    print(f"Max images:              {args.max_img}")
    print(f"Device:                  {args.device}")
    print(f"IoU threshold :          {args.iou_thr}")
    print(f"Conf-thr:                {args.conf_thr}")
    print(f"Train:                   {args.train}")
    print("===================================\n")

    if args.train:
        model_dir = train_model()
    else:
        model_dir = args.model

    fgsm_attack(model_dir, args.img_dir, args.labels_dir, args.img_out, args.pert_with_eps_dir,
                args.pert_dir, args.eps, args.max_img, args.device)

    generate_cm(args.img_out, args.img_dir, args.labels_dir, model_dir, args.device, args.cm_out, args.iou_thr,
                args.conf_thr)


if __name__ == "__main__":
    main()
