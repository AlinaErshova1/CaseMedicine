"""
Сравнение двух U-Net моделей на одних снимках.
Автоматически подбирает архитектуру для каждой модели
(поддерживает legacy v1 и v2).

Запуск:
    python -m src.seg_compare --image data/test/meningioma/1458.jpg --ckpt_a checkpoints/best_unet_v1_old.pth --ckpt_b checkpoints/best_unet.pth
    python -m src.seg_compare --batch data/test --per_class 2 --ckpt_a checkpoints/best_unet_v1_old.pth --ckpt_b checkpoints/best_unet.pth --save_dir checkpoints/compare
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .seg_model import build_unet_from_ckpt
from .seg_dataset import IMAGENET_MEAN, IMAGENET_STD


def load_unet(ckpt_path, device):
    """Загружает U-Net, автоматически определяя архитектуру из чекпоинта."""
    ckpt = torch.load(ckpt_path, map_location=device)
    img_size = ckpt.get("img_size", 256)
    model = build_unet_from_ckpt(ckpt, device=device)
    meta = ckpt.get("metrics", {})
    meta["epoch"] = ckpt.get("epoch", "?")
    meta["architecture"] = ckpt.get("architecture", "v2?")
    return model, img_size, meta


@torch.no_grad()
def predict_mask(model, img_rgb, img_size, device):
    """Возвращает вероятность (H, W) в исходном размере."""
    orig_h, orig_w = img_rgb.shape[:2]
    x = cv2.resize(img_rgb, (img_size, img_size)).astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    logits = model(x)
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    probs = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return probs


def overlay_single(img_rgb, mask_bin, color, alpha=0.5):
    overlay = img_rgb.copy()
    overlay[mask_bin > 0] = color
    return cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)


def overlay_two(img_rgb, mask_a, mask_b, alpha=0.45):
    """Красный = только A, синий = только B, фиолетовый = пересечение."""
    overlay = img_rgb.copy()
    only_a = (mask_a > 0) & (mask_b == 0)
    only_b = (mask_b > 0) & (mask_a == 0)
    both = (mask_a > 0) & (mask_b > 0)
    overlay[only_a] = [255, 60, 60]
    overlay[only_b] = [60, 100, 255]
    overlay[both] = [180, 60, 200]
    return cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)


def compare_single(img_path, model_a, model_b, size_a, size_b,
                    meta_a, meta_b, device, threshold=0.75,
                    save=None, name_a="A", name_b="B"):
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Не читается: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    prob_a = predict_mask(model_a, img_rgb, size_a, device)
    prob_b = predict_mask(model_b, img_rgb, size_b, device)

    mask_a = (prob_a > threshold).astype(np.uint8)
    mask_b = (prob_b > threshold).astype(np.uint8)

    inter = ((mask_a > 0) & (mask_b > 0)).sum()
    union = ((mask_a > 0) | (mask_b > 0)).sum()
    iou_ab = inter / max(union, 1)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Оригинал\n{Path(img_path).name}", fontsize=11)

    axes[1].imshow(prob_a, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(
        f"{name_a}\nDice={meta_a.get('dice', 0):.3f} | "
        f"area={mask_a.mean() * 100:.1f}%",
        fontsize=11,
    )

    axes[2].imshow(prob_b, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title(
        f"{name_b}\nDice={meta_b.get('dice', 0):.3f} | "
        f"area={mask_b.mean() * 100:.1f}%",
        fontsize=11,
    )

    axes[3].imshow(overlay_single(img_rgb, mask_a, [255, 0, 0]))
    axes[3].set_title(f"{name_a} overlay (красный)", fontsize=11)

    axes[4].imshow(overlay_two(img_rgb, mask_a, mask_b))
    axes[4].set_title(
        f"Сравнение\nIoU A↔B = {iou_ab:.3f}\n"
        f"красный={name_a}  синий={name_b}  фиолет.=совпадение",
        fontsize=11,
    )

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  → {save}")
    else:
        plt.show()

    return {
        "image": str(img_path),
        "area_a_pct": float(mask_a.mean() * 100),
        "area_b_pct": float(mask_b.mean() * 100),
        "iou_ab": float(iou_ab),
    }


def compare_batch(root_dir, model_a, model_b, size_a, size_b,
                    meta_a, meta_b, device, per_class=3, threshold=0.75,
                    save_dir="checkpoints/compare", name_a="A", name_b="B"):
    root = Path(root_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if not subdirs:
        subdirs = [root]

    all_stats = []

    for cls_dir in sorted(subdirs):
        files = sorted(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")))
        files = files[:per_class]

        for i, f in enumerate(files):
            out = save_dir / f"{cls_dir.name}_{i:02d}.png"
            print(f"[INFO] {f.name}")
            try:
                stats = compare_single(
                    f, model_a, model_b, size_a, size_b,
                    meta_a, meta_b, device, threshold=threshold,
                    save=str(out), name_a=name_a, name_b=name_b,
                )
                stats["class"] = cls_dir.name
                all_stats.append(stats)
            except Exception as e:
                print(f"  [WARN] Ошибка: {e}")

    # Сводка
    if all_stats:
        print("\n" + "=" * 60)
        print("  СВОДКА")
        print("=" * 60)
        for s in all_stats:
            print(f"  {s['class']:12s} | {Path(s['image']).name:25s} | "
                  f"area_a={s['area_a_pct']:5.1f}% | "
                  f"area_b={s['area_b_pct']:5.1f}% | "
                  f"IoU={s['iou_ab']:.3f}")

        avg_a = np.mean([s["area_a_pct"] for s in all_stats])
        avg_b = np.mean([s["area_b_pct"] for s in all_stats])
        avg_iou = np.mean([s["iou_ab"] for s in all_stats])
        print("=" * 60)
        print(f"  Среднее: {name_a}={avg_a:.1f}% | "
              f"{name_b}={avg_b:.1f}% | IoU A↔B={avg_iou:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--batch", type=str, default=None)
    parser.add_argument("--per_class", type=int, default=3)
    parser.add_argument("--save_dir", type=str,
                        default="checkpoints/compare_batch")

    parser.add_argument("--ckpt_a", type=str, required=True)
    parser.add_argument("--ckpt_b", type=str, required=True)
    parser.add_argument("--name_a", type=str, default="Model A")
    parser.add_argument("--name_b", type=str, default="Model B")

    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    if not args.image and not args.batch:
        parser.error("Укажите --image или --batch")

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    print(f"[INFO] Device: {device}")

    print(f"\n[INFO] Загружаю A: {args.ckpt_a}")
    model_a, size_a, meta_a = load_unet(args.ckpt_a, device)
    print(f"  → img_size={size_a}, dice={meta_a.get('dice', 0):.4f}, "
          f"epoch={meta_a.get('epoch')}")

    print(f"\n[INFO] Загружаю B: {args.ckpt_b}")
    model_b, size_b, meta_b = load_unet(args.ckpt_b, device)
    print(f"  → img_size={size_b}, dice={meta_b.get('dice', 0):.4f}, "
          f"epoch={meta_b.get('epoch')}")

    if args.image:
        compare_single(
            args.image, model_a, model_b, size_a, size_b,
            meta_a, meta_b, device, threshold=args.threshold,
            save=args.save, name_a=args.name_a, name_b=args.name_b,
        )
    else:
        compare_batch(
            args.batch, model_a, model_b, size_a, size_b,
            meta_a, meta_b, device, per_class=args.per_class,
            threshold=args.threshold, save_dir=args.save_dir,
            name_a=args.name_a, name_b=args.name_b,
        )


if __name__ == "__main__":
    main()