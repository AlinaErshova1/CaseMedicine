"""
Обучение U-Net для сегментации опухолей.

Улучшения:
  - Заморозка энкодера на первые N эпох (для малых датасетов)
  - TTA-валидация (усреднение масок по 4 аугментациям)
  - Threshold sweep для поиска оптимального порога
  - Два LR для энкодера и декодера
  - Cosine annealing с warmup

Запуск:
    python -m src.seg_train --epochs 60 --batch_size 4 --img_size 256 --device cpu
"""
import matplotlib
matplotlib.use("Agg")
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .seg_dataset import (SegmentationDataset, find_pairs,
                          get_seg_transforms, get_tta_transforms)
from .seg_model import build_unet
from .seg_losses import DiceBCELoss, FocalDiceLoss, TverskyBCELoss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ---------- Метрики ----------

def compute_dice_iou(preds, targets, eps=1e-6):
    """preds, targets: бинарные тензоры (B, 1, H, W)."""
    inter = (preds * targets).sum().item()
    pred_sum = preds.sum().item()
    gt_sum = targets.sum().item()
    union = ((preds + targets) > 0).float().sum().item()
    dice = (2 * inter + eps) / (pred_sum + gt_sum + eps)
    iou = (inter + eps) / (union + eps)
    return dice, iou


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = 0

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, masks)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        d, i = compute_dice_iou(preds, masks)
        total_dice += d
        total_iou += i
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": total_dice / max(n_batches, 1),
        "iou": total_iou / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate_tta(model, dataset, criterion, device, threshold=0.5,
                 tta_transforms_list=None):
    """
    TTA-валидация: усредняем вероятности по нескольким трансформам.
    dataset.samples должен быть в виде [(img_path, mask_path), ...].
    """
    from torch.utils.data import DataLoader
    from .seg_dataset import SegmentationDataset

    if tta_transforms_list is None:
        tta_transforms_list = get_tta_transforms()

    model.eval()
    n = len(dataset)
    prob_sum = np.zeros((n, 1, 256, 256), dtype=np.float32)  # предположим 256
    masks_all = None

    for t_idx, tf in enumerate(tta_transforms_list):
        ds = SegmentationDataset(dataset.pairs, transform=tf)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

        all_probs = []
        all_masks = []
        for imgs, masks in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            if masks_all is None:
                all_masks.append(masks.numpy())

        probs_cat = np.concatenate(all_probs, axis=0)

        # Приводим к одной форме
        if prob_sum.shape[1:] != probs_cat.shape[1:]:
            prob_sum = np.zeros_like(probs_cat)

        prob_sum += probs_cat

        if masks_all is None:
            masks_all = np.concatenate(all_masks, axis=0)

    prob_avg = prob_sum / len(tta_transforms_list)
    preds = (prob_avg > threshold).astype(np.float32)
    targets = masks_all.astype(np.float32)

    inter = (preds * targets).sum()
    pred_sum = preds.sum()
    gt_sum = targets.sum()
    union = ((preds + targets) > 0).sum()

    dice = (2 * inter + 1e-6) / (pred_sum + gt_sum + 1e-6)
    iou = (inter + 1e-6) / (union + 1e-6)
    return {"dice": float(dice), "iou": float(iou)}


# ---------- Обучение ----------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in tqdm(loader, desc="train", leave=False, ncols=80):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def find_best_threshold(model, loader, device):
    """Пороговый sweep на валидации — ищем порог с максимальным Dice."""
    model.eval()
    all_probs = []
    all_masks = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_masks.append(masks.numpy())

    probs = np.concatenate(all_probs, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    best_t, best_d = 0.5, 0.0
    for t in np.arange(0.2, 0.85, 0.05):
        preds = (probs > t).astype(np.float32)
        inter = (preds * masks).sum()
        denom = preds.sum() + masks.sum()
        dice = (2 * inter + 1e-6) / (denom + 1e-6)
        if dice > best_d:
            best_d = dice
            best_t = t
    return best_t, best_d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="data/train")
    parser.add_argument("--masks_dir", type=str, default="data/masks/train")
    parser.add_argument("--class_names", type=str,
                        default="glioma,meningioma,pituitary")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr_encoder", type=float, default=1e-5,
                        help="Меньше, чем декодер — энкодер уже предобучен")
    parser.add_argument("--lr_decoder", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--loss", type=str, default="dice_bce",
                        choices=["dice_bce", "focal_dice", "tversky_bce"])
    parser.add_argument("--freeze_epochs", type=int, default=10,
                        help="Сколько эпох энкодер заморожен")
    parser.add_argument("--classifier_ckpt", type=str,
                        default="checkpoints/best_efficientnet_b0.pth")
    parser.add_argument("--no_pretrained_encoder", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output", type=str,
                        default="checkpoints/best_unet.pth")
    parser.add_argument("--log_dir", type=str, default="logs/unet")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    print(f"[INFO] Device: {device}")

    class_names = [c.strip() for c in args.class_names.split(",")]
    print(f"[INFO] Классы: {class_names}")

    # --- Пары ---
    pairs, counts = find_pairs(args.images_dir, args.masks_dir, class_names)
    print(f"[INFO] Всего пар: {len(pairs)}")
    for cls, n in counts.items():
        print(f"  {cls}: {n}")

    if len(pairs) < 10:
        raise SystemExit(f"Слишком мало пар: {len(pairs)}")

    train_pairs, val_pairs = train_test_split(
        pairs, test_size=args.val_ratio, random_state=args.seed, shuffle=True
    )
    print(f"[INFO] Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    train_ds = SegmentationDataset(
        train_pairs, transform=get_seg_transforms(args.img_size, is_train=True)
    )
    val_ds = SegmentationDataset(
        val_pairs, transform=get_seg_transforms(args.img_size, is_train=False)
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # --- Модель ---
    ckpt_path = None if args.no_pretrained_encoder else args.classifier_ckpt
    model = build_unet(pretrained=True, classifier_ckpt=ckpt_path)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Параметров: {n_params / 1e6:.2f}M")

    # --- Заморозка энкодера ---
    if args.freeze_epochs > 0:
        model.freeze_encoder(True)
        print(f"[INFO] Энкодер заморожен на {args.freeze_epochs} эпох")

    # --- Loss ---
    if args.loss == "focal_dice":
        criterion = FocalDiceLoss()
    elif args.loss == "tversky_bce":
        criterion = TverskyBCELoss(alpha=0.3, beta=0.7)
    else:
        criterion = DiceBCELoss()
    print(f"[INFO] Loss: {type(criterion).__name__}")

    # --- Оптимизатор: два LR ---
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = [p for n, p in model.named_parameters()
                      if not n.startswith("encoder.")]

    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": args.lr_encoder},
        {"params": decoder_params, "lr": args.lr_decoder},
    ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    log_dir = Path(args.log_dir) / time.strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_dice = 0.0
    patience_counter = 0
    history = []

    print(f"\n{'=' * 70}")
    print(f"Начинаю обучение: {args.epochs} эпох, batch={args.batch_size}, "
          f"img={args.img_size}")
    print(f"{'=' * 70}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Разморозка энкодера после freeze_epochs
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            model.freeze_encoder(False)
            print(f"[INFO] Эпоха {epoch}: энкодер разморожен")

        train_loss = train_one_epoch(model, train_loader, criterion,
                                      optimizer, device)
        metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr_enc = optimizer.param_groups[0]["lr"]
        lr_dec = optimizer.param_groups[1]["lr"]
        dt = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={metrics['loss']:.4f} | "
              f"dice={metrics['dice']:.4f} | iou={metrics['iou']:.4f} | "
              f"lr_enc={lr_enc:.1e} lr_dec={lr_dec:.1e} | {dt:.1f}s")

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", metrics["loss"], epoch)
        writer.add_scalar("dice/val", metrics["dice"], epoch)
        writer.add_scalar("iou/val", metrics["iou"], epoch)

        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "img_size": args.img_size,
                "class_names": class_names,
                "metrics": metrics,
                "epoch": epoch,
            }, out_path)
            print(f"  ↳ saved {out_path} (dice={best_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[INFO] Early stopping на эпохе {epoch}")
                break

    writer.close()

    # --- Финальная оценка ---
    print("\n" + "=" * 70)
    print("ФИНАЛЬНАЯ ОЦЕНКА")
    print("=" * 70)
    best = torch.load(out_path, map_location=device)
    model.load_state_dict(best["model_state"])

    metrics = evaluate(model, val_loader, criterion, device)
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Dice: {metrics['dice']:.4f}")
    print(f"IoU:  {metrics['iou']:.4f}")

    # Пороговый sweep
    best_t, dice_at_best_t = find_best_threshold(model, val_loader, device)
    print(f"\nОптимальный порог: {best_t:.2f} (Dice={dice_at_best_t:.4f})")

    # TTA (опционально, может быть медленно на CPU)
    print("\n[INFO] TTA-оценка (может занять время)...")
    try:
        tta_metrics = evaluate_tta(model, val_ds, criterion, device,
                                    threshold=best_t)
        print(f"TTA Dice: {tta_metrics['dice']:.4f}")
        print(f"TTA IoU:  {tta_metrics['iou']:.4f}")
    except Exception as e:
        print(f"[WARN] TTA не удался: {e}")
        tta_metrics = {"dice": None, "iou": None}

    # Сохраняем метаданные
    meta = {
        "history": history,
        "best_metrics": metrics,
        "best_threshold": float(best_t),
        "tta_metrics": tta_metrics,
        "config": vars(args),
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n[DONE] Лучший Dice = {best_dice:.4f}")
    print(f"[DONE] Веса: {out_path}")
    print(f"[DONE] Threshold: {best_t:.2f}")
    print(f"[DONE] TensorBoard: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()