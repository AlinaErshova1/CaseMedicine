"""
Предсказание класса опухоли на одном изображении.

Примеры:
    # Просто предсказание
    python -m src.predict --image data/test/glioma/Te-gl_0010.png

    # С Grad-CAM (покажет, куда смотрит модель)
    python -m src.predict --image data/test/glioma/Te-gl_0010.png --gradcam

    # С TTA (усреднение по аугментациям, точнее)
    python -m src.predict --image data/test/glioma/Te-gl_0010.png --tta

    # Сохранить результат
    python -m src.predict --image data/test/glioma/Te-gl_0010.png --gradcam --save result.png
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from .config import CFG
from .model import build_model
from .transforms import get_val_transforms, get_tta_transforms


DEFAULT_CKPT = "checkpoints/best_efficientnet_b0.pth"


def load_model(ckpt_path, device):
    """Загружает модель и метаданные из чекпоинта."""
    ckpt = torch.load(ckpt_path, map_location=device)
    class_names = ckpt["class_names"]
    backbone = ckpt["backbone"]
    img_size = ckpt.get("img_size", CFG.img_size)

    model, _, _ = build_model(backbone, len(class_names), CFG.dropout)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    meta = {
        "class_names": class_names,
        "backbone": backbone,
        "img_size": img_size,
        "metrics": ckpt.get("metrics", {}),
        "epoch": ckpt.get("epoch", None),
    }
    return model, meta


@torch.no_grad()
def predict_simple(model, img_path, img_size, class_names, device):
    """Предсказание без TTA."""
    img = np.array(Image.open(img_path).convert("RGB"))
    tensor = get_val_transforms(img_size)(image=img)["image"].unsqueeze(0).to(device)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs, img


@torch.no_grad()
def predict_tta(model, img_path, img_size, class_names, device, n_tta=4):
    """Предсказание с TTA — усредняем softmax по нескольким аугментациям."""
    img = np.array(Image.open(img_path).convert("RGB"))
    tta_transforms = get_tta_transforms(img_size)[:n_tta]
    probs_sum = np.zeros(len(class_names), dtype=np.float64)
    for tf in tta_transforms:
        tensor = tf(image=img)["image"].unsqueeze(0).to(device)
        probs_sum += F.softmax(model(tensor), dim=1).cpu().numpy()[0]
    return probs_sum / len(tta_transforms), img


def predict_gradcam(model, img_path, img_size, class_names, device,
                    target_class=None):
    """Предсказание + Grad-CAM тепловая карта."""
    from .gradcam_utils import GradCAM, get_target_layer, overlay_heatmap

    img = np.array(Image.open(img_path).convert("RGB"))
    tensor = get_val_transforms(img_size)(image=img)["image"].unsqueeze(0).to(device)

    # Определяем бэкбон по имени модуля модели
    backbone_name = None
    for name in ("efficientnet_b0", "resnet50", "convnext_tiny"):
        if name in str(type(model)).lower() or hasattr(model, "features"):
            backbone_name = "efficientnet_b0"  # дефолт; уточним ниже
            break

    # Универсальный выбор target layer — последний Conv2d
    target_layer = None
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break
    if target_layer is None:
        raise RuntimeError("Не найден свёрточный слой для Grad-CAM")

    cam = GradCAM(model, target_layer)
    heatmap, pred_class, probs = cam.generate(tensor, target_class=target_class)
    overlay = overlay_heatmap(img, heatmap, alpha=0.45)
    return probs, img, heatmap, overlay, pred_class


def print_report(probs, class_names, img_path, tta_used=False):
    """Красивый вывод результата в консоль."""
    pred = int(np.argmax(probs))
    print("\n" + "=" * 55)
    print(f"  Изображение: {img_path}")
    print(f"  TTA: {'включено' if tta_used else 'выключено'}")
    print("=" * 55)
    print(f"  ПРЕДСКАЗАНИЕ: {class_names[pred].upper()}")
    print(f"  Уверенность:  {probs[pred] * 100:.2f}%")
    print("-" * 55)
    print("  Распределение вероятностей:")
    for name, p in sorted(zip(class_names, probs), key=lambda x: -x[1]):
        bar = "█" * int(p * 40)
        marker = " ←" if name == class_names[pred] else ""
        print(f"    {name:12s} {p * 100:6.2f}%  {bar}{marker}")
    print("=" * 55 + "\n")


def save_visualization(img, heatmap, overlay, probs, class_names,
                        pred_class, save_path):
    """Сохраняет 4 панели: оригинал, heatmap, overlay, bar-chart вероятностей."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Оригинал", fontsize=13)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(heatmap, cmap="jet")
    axes[0, 1].set_title("Grad-CAM (красное = важно для модели)", fontsize=13)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title(
        f"Overlay — {class_names[pred_class]} "
        f"({probs[pred_class] * 100:.1f}%)",
        fontsize=13,
    )
    axes[1, 0].axis("off")

    # Bar chart
    colors = ["#cccccc"] * len(class_names)
    colors[pred_class] = "#e74c3c"
    bars = axes[1, 1].barh(class_names, probs * 100, color=colors)
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].set_xlabel("Вероятность, %", fontsize=12)
    axes[1, 1].set_title("Распределение вероятностей", fontsize=13)
    axes[1, 1].invert_yaxis()
    for bar, p in zip(bars, probs):
        axes[1, 1].text(p * 100 + 1.5, bar.get_y() + bar.get_height() / 2,
                        f"{p * 100:.1f}%", va="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Сохранено: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="Путь к изображению МРТ")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                        help="Путь к чекпоинту")
    parser.add_argument("--tta", action="store_true",
                        help="Использовать Test-Time Augmentation")
    parser.add_argument("--tta_n", type=int, default=4,
                        help="Сколько TTA-аугментаций")
    parser.add_argument("--gradcam", action="store_true",
                        help="Построить Grad-CAM")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Класс для Grad-CAM (по умолчанию = предсказанный)")
    parser.add_argument("--save", type=str, default=None,
                        help="Куда сохранить визуализацию")
    parser.add_argument("--device", type=str, default=CFG.device)
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading: {args.ckpt}")

    model, meta = load_model(args.ckpt, device)
    class_names = meta["class_names"]
    img_size = meta["img_size"]
    print(f"[INFO] Backbone: {meta['backbone']} | Classes: {class_names}")
    if meta["metrics"]:
        m = meta["metrics"]
        print(f"[INFO] Чекпоинт: epoch={meta['epoch']} | "
              f"acc={m.get('acc', '?'):.4f} | "
              f"recall_tumor={m.get('recall_tumor', '?'):.4f}")

    # ---- Инференс ----
    heatmap = overlay = None
    pred_class = None

    if args.gradcam:
        probs, img, heatmap, overlay, pred_class = predict_gradcam(
            model, args.image, img_size, class_names, device,
            target_class=args.target_class,
        )
    elif args.tta:
        probs, img = predict_tta(model, args.image, img_size,
                                  class_names, device, n_tta=args.tta_n)
    else:
        probs, img = predict_simple(model, args.image, img_size,
                                     class_names, device)

    print_report(probs, class_names, args.image, tta_used=args.tta)

    # ---- Визуализация ----
    if args.gradcam and args.save:
        save_visualization(img, heatmap, overlay, probs, class_names,
                           pred_class, args.save)
    elif args.save:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap="gray")
        pred = int(np.argmax(probs))
        ax.set_title(
            f"Pred: {class_names[pred]} ({probs[pred] * 100:.1f}%)",
            fontsize=12,
        )
        ax.axis("off")
        plt.savefig(args.save, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[DONE] Сохранено: {args.save}")
    elif args.gradcam:
        plt.show()


if __name__ == "__main__":
    main()