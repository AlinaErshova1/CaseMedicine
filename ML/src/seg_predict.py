"""
Инференс обученной U-Net.
Показывает: оригинал / предсказанную маску / наложение.

Запуск:
    python -m src.seg_predict --image data/train/glioma/001.png
    python -m src.seg_predict --image data/train/glioma/001.png --save out.png
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

from .seg_model import build_unet
from .seg_dataset import IMAGENET_MEAN, IMAGENET_STD


@torch.no_grad()
def predict_mask(model, img_path, img_size, device, threshold=0.5):
    """Возвращает оригинал (RGB) и предсказанную бинарную маску."""
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Не читается: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_shape = img_rgb.shape[:2]

    # Предобработка
    x = cv2.resize(img_rgb, (img_size, img_size)).astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Инференс
    logits = model(x)
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # Возвращаем к оригинальному размеру
    mask = cv2.resize(probs, (orig_shape[1], orig_shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    mask_binary = (mask > threshold).astype(np.uint8)

    return img_rgb, mask, mask_binary


def visualize(img_rgb, mask_prob, mask_bin, save_to=None, show=True):
    """4 панели: оригинал / вероятность / бинарная маска / наложение."""
    overlay = img_rgb.copy()
    overlay[mask_bin > 0] = [255, 0, 0]
    blended = cv2.addWeighted(img_rgb, 0.55, overlay, 0.45, 0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Оригинал")
    axes[1].imshow(mask_prob, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Вероятность (heatmap)")
    axes[2].imshow(mask_bin, cmap="gray")
    axes[2].set_title(f"Маска (площадь {mask_bin.mean() * 100:.1f}%)")
    axes[3].imshow(blended)
    axes[3].set_title("Наложение (красное = опухоль)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, dpi=100, bbox_inches="tight")
        plt.close()
    elif show:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_unet.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Порог бинаризации маски (0.5 по умолчанию)")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    print(f"[INFO] Device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    img_size = ckpt.get("img_size", 256)
    print(f"[INFO] Model epoch={ckpt.get('epoch')}, "
          f"metrics={ckpt.get('metrics')}")

    model = build_unet(pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    print(f"[INFO] Predicting {args.image}")
    img_rgb, mask_prob, mask_bin = predict_mask(
        model, args.image, img_size, device, threshold=args.threshold
    )

    area_px = int(mask_bin.sum())
    total_px = mask_bin.size
    print(f"\n[РЕЗУЛЬТАТ]")
    print(f"  Площадь опухоли: {area_px} пикселей "
          f"({area_px / total_px * 100:.2f}%)")
    print(f"  Размер снимка: {img_rgb.shape}")

    visualize(img_rgb, mask_prob, mask_bin,
              save_to=args.save, show=not args.no_show)
    if args.save:
        print(f"[DONE] Сохранено: {args.save}")


if __name__ == "__main__":
    main()