"""
CLI для визуализации Grad-CAM на одном или нескольких изображениях.

Примеры:
    # Одно изображение
    python -m src.visualize --ckpt checkpoints/best_efficientnet_b0.pth \
        --image data/test/glioma/Te-gl_0010.png

    # С bbox вокруг зоны интереса
    python -m src.visualize --ckpt checkpoints/best_efficientnet_b0.pth \
        --image data/test/glioma/Te-gl_0010.png --bbox --save out.png

    # Батч: по одному примеру каждого класса из data/test
    python -m src.visualize --ckpt checkpoints/best_efficientnet_b0.pth \
        --batch data/test --per_class 3 --save_dir checkpoints/gradcam
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from .config import CFG
from .model import build_model
from .gradcam_utils import (explain_single, cam_to_bbox, overlay_heatmap,
                            GradCAM, get_target_layer, _load_image)


def draw_bbox_on_image(img_rgb, bbox, color=(255, 0, 0), thickness=3):
    img = img_rgb.copy()
    if bbox is None:
        return img
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def run_single(args, model, class_names, device):
    print(f"[INFO] Обрабатываю: {args.image}")
    result = explain_single(
        model, args.image, class_names, args.backbone, device,
        target_class=args.target_class,
        save_path=args.save,
        show=not args.no_show,
    )
    print(f"\nПредсказание: {result['predicted_class']}")
    for cls, p in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        print(f"  {cls:12s}: {p*100:6.2f}%")

    if args.bbox:
        # Повторно получаем изображение и строим bbox
        img_rgb, tensor = _load_image(args.image, CFG.img_size)
        tensor = tensor.to(device)
        cam = GradCAM(model, get_target_layer(model, args.backbone))
        heatmap, _, _ = cam.generate(tensor, target_class=args.target_class)
        bbox = cam_to_bbox(heatmap, img_rgb.shape, threshold=args.bbox_threshold)

        if bbox is not None:
            img_with_bbox = draw_bbox_on_image(img_rgb, bbox)
            plt.figure(figsize=(6, 6))
            plt.imshow(img_with_bbox)
            plt.title(f"Область опухоли (bbox)")
            plt.axis("off")
            bbox_path = None
            if args.save:
                bbox_path = str(Path(args.save).with_name(
                    Path(args.save).stem + "_bbox" + Path(args.save).suffix))
                plt.savefig(bbox_path, dpi=120, bbox_inches="tight")
            plt.show()
            print(f"[DONE] Bbox сохранён: {bbox_path}")
        else:
            print("[WARN] Heatmap ниже порога — bbox не построен")


def run_batch(args, model, class_names, device):
    root = Path(args.batch)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for cls in class_names:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
        files = sorted(list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")))[:args.per_class]
        for i, f in enumerate(files):
            out = save_dir / f"{cls}_{i:02d}.png"
            print(f"[INFO] {f.name} → {out.name}")
            explain_single(
                model, str(f), class_names, args.backbone, device,
                target_class=None, save_path=str(out), show=False,
            )
    print(f"\n[DONE] Все визуализации в {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image", type=str, default=None,
                        help="Путь к одному изображению")
    parser.add_argument("--batch", type=str, default=None,
                        help="Папка с классами (для батч-визуализации)")
    parser.add_argument("--per_class", type=int, default=3,
                        help="Сколько примеров на класс в батч-режиме")
    parser.add_argument("--save_dir", type=str, default="checkpoints/gradcam")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Класс для объяснения (по умолчанию = argmax)")
    parser.add_argument("--bbox", action="store_true",
                        help="Построить bbox вокруг зоны интереса")
    parser.add_argument("--bbox_threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=CFG.device)
    parser.add_argument("--save", type=str, default=None,
                        help="Куда сохранить визуализацию")
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    if not args.image and not args.batch:
        parser.error("Нужно указать --image или --batch")

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    ckpt = torch.load(args.ckpt, map_location=device)
    class_names = ckpt["class_names"]
    args.backbone = ckpt["backbone"]

    model, _, _ = build_model(args.backbone, len(class_names), CFG.dropout)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    if args.image:
        run_single(args, model, class_names, device)
    else:
        run_batch(args, model, class_names, device)


if __name__ == "__main__":
    main()