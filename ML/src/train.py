import argparse
import json
import random
import time
from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             recall_score, classification_report,
                             confusion_matrix)
from tqdm import tqdm

from .config import CFG
from .dataset import (build_dataloaders, compute_class_weights,
                      stratified_split)
from .model import build_model
from .losses import build_criterion


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    preds = all_probs.argmax(axis=1)

    # per-class recall
    recalls = recall_score(all_labels, preds, average=None,
                           labels=list(range(cfg.num_classes)),
                           zero_division=0)
    tumor_recall = float(np.mean([recalls[i] for i in cfg.tumor_class_indices]))

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "acc": accuracy_score(all_labels, preds),
        "f1_macro": f1_score(all_labels, preds, average="macro", zero_division=0),
        "recall_tumor": tumor_recall,
    }
    for i, name in enumerate(cfg.class_names):
        metrics[f"recall_{name}"] = float(recalls[i])

    try:
        metrics["auc_macro"] = roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        metrics["auc_macro"] = float("nan")

    return metrics, all_labels, preds, all_probs


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Корень с train/ и test/")
    parser.add_argument("--single_dir", type=str, default=None,
                        help="Одна папка с классами — split автоматически")
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--backbone", type=str, default=CFG.backbone)
    parser.add_argument("--device", type=str, default=CFG.device)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--loss", type=str, default=CFG.loss_type,
                        choices=["ce", "focal"])
    parser.add_argument("--monitor", type=str, default=CFG.monitor_metric,
                        choices=["recall_tumor", "f1_macro", "acc"])
    args = parser.parse_args()

    cfg = CFG
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.backbone = args.backbone
    cfg.device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg.loss_type = args.loss
    cfg.monitor_metric = args.monitor

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    print(f"[INFO] Device: {device} | Backbone: {cfg.backbone} | Loss: {cfg.loss_type}")
    print(f"[INFO] Monitor metric: {cfg.monitor_metric}")

    # ---- Данные ----
    if args.single_dir:
        single_dir = Path(args.single_dir)
        cfg.train_dir = single_dir
        train_samples, val_samples = stratified_split(
            single_dir, cfg.class_names, val_ratio=args.val_ratio, seed=cfg.seed
        )
        print(f"[INFO] Split: train={len(train_samples)}, val={len(val_samples)}")
        train_loader, val_loader, train_ds, val_ds = build_dataloaders(
            cfg, train_samples=train_samples, val_samples=val_samples,
            train_dir=single_dir, val_dir=single_dir,
        )
    else:
        if args.data_dir:
            cfg.train_dir = Path(args.data_dir) / "train"
            cfg.val_dir = Path(args.data_dir) / "test"
        train_loader, val_loader, train_ds, val_ds = build_dataloaders(cfg)

    print(f"[INFO] Train по классам: {train_ds.class_counts()}")
    print(f"[INFO] Val   по классам: {val_ds.class_counts()}")

    # ---- Модель ----
    model, backbone_params, head_params = build_model(
        cfg.backbone, cfg.num_classes, cfg.dropout
    )
    model.to(device)

    # ---- Class weights + criterion ----
    weights = None
    if cfg.use_class_weights:
        weights = compute_class_weights(train_ds.samples, cfg.num_classes)
        print(f"[INFO] Class weights: {weights.numpy().round(3)}")
    criterion = build_criterion(cfg, class_weights=weights, device=device)
    print(f"[INFO] Criterion: {type(criterion).__name__}")

    # ---- Оптимизатор ----
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": cfg.lr_backbone},
        {"params": head_params, "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    writer = SummaryWriter(log_dir=str(cfg.log_dir / time.strftime("%Y%m%d_%H%M%S")))

    # ---- Цикл обучения ----
    best_score = -1.0
    patience = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics, y_true, y_pred, y_prob = evaluate(model, val_loader, criterion,
                                                   device, cfg)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={metrics['loss']:.4f} | "
            f"acc={metrics['acc']:.4f} | f1={metrics['f1_macro']:.4f} | "
            f"recall_tumor={metrics['recall_tumor']:.4f} | "
            f"auc={metrics['auc_macro']:.4f} | lr={lr_now:.2e} | {time.time()-t0:.1f}s"
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", metrics["loss"], epoch)
        writer.add_scalar("acc/val", metrics["acc"], epoch)
        writer.add_scalar("f1/val", metrics["f1_macro"], epoch)
        writer.add_scalar("recall_tumor/val", metrics["recall_tumor"], epoch)
        for name in cfg.class_names:
            writer.add_scalar(f"recall/{name}", metrics[f"recall_{name}"], epoch)

        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        # ---- Сохранение лучшей модели по monitor_metric ----
        score = metrics[cfg.monitor_metric]
        if score > best_score:
            best_score = score
            patience = 0
            ckpt_path = cfg.checkpoint_dir / f"best_{cfg.backbone}.pth"
            torch.save({
                "model_state": model.state_dict(),
                "class_names": cfg.class_names,
                "backbone": cfg.backbone,
                "img_size": cfg.img_size,
                "metrics": metrics,
                "epoch": epoch,
                "monitor_metric": cfg.monitor_metric,
            }, ckpt_path)
            print(f"  ↳ saved {ckpt_path} ({cfg.monitor_metric}={best_score:.4f})")
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                print(f"[INFO] Early stopping на эпохе {epoch}")
                break

    writer.close()

    # ---- Финальная оценка ----
    print("\n=== Финальная оценка лучшей модели ===")
    best_ckpt = torch.load(cfg.checkpoint_dir / f"best_{cfg.backbone}.pth",
                           map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    metrics, y_true, y_pred, y_prob = evaluate(model, val_loader, criterion,
                                               device, cfg)

    print(classification_report(y_true, y_pred, target_names=cfg.class_names,
                                digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nRecall опухолевых классов (avg): {metrics['recall_tumor']:.4f}")

    with open(cfg.checkpoint_dir / f"history_{cfg.backbone}.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[DONE] Лучший {cfg.monitor_metric} = {best_score:.4f}")
    print(f"[DONE] Веса: {cfg.checkpoint_dir / f'best_{cfg.backbone}.pth'}")
    print(f"[DONE] TensorBoard: tensorboard --logdir {cfg.log_dir}")


if __name__ == "__main__":
    main()