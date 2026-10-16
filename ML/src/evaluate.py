import argparse
from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             recall_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize

from .config import CFG
from .dataset import build_dataloaders, BrainTumorDataset
from .model import build_model
from .transforms import get_val_transforms, get_tta_transforms

from torch.utils.data import DataLoader


@torch.no_grad()
def run_inference_plain(model, loader, device):
    model.eval()
    probs, labels = [], []
    for imgs, y in loader:
        imgs = imgs.to(device)
        out = torch.softmax(model(imgs), dim=1)
        probs.append(out.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


@torch.no_grad()
def run_inference_tta(model, dataset, device, cfg, batch_size=32):
    """
    TTA: прогоняем датасет через N трансформов, усредняем softmax.
    Ожидает dataset.samples — список (path, label).
    """
    tta_transforms = get_tta_transforms(cfg.img_size)[: cfg.tta_n]
    num = len(dataset.samples)
    n_classes = cfg.num_classes
    prob_sum = np.zeros((num, n_classes), dtype=np.float64)
    labels = np.array([y for _, y in dataset.samples])

    for t_idx, tf in enumerate(tta_transforms):
        # Создаём датасет с этим TTA-трансформом, используя те же samples
        tta_ds = BrainTumorDataset(dataset.root_dir, dataset.class_names,
                                   transform=tf, samples=dataset.samples)
        loader = DataLoader(tta_ds, batch_size=batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
        model.eval()
        all_probs = []
        for imgs, _ in loader:
            imgs = imgs.to(device)
            out = torch.softmax(model(imgs), dim=1).cpu().numpy()
            all_probs.append(out)
        all_probs = np.concatenate(all_probs, axis=0)
        # Важно: tta_ds сортирует samples по пути — совпадает с dataset.samples
        prob_sum += all_probs
        print(f"  TTA {t_idx + 1}/{len(tta_transforms)} done")

    return prob_sum / len(tta_transforms), labels


def plot_confusion(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_roc(y_true, y_prob, class_names, out_path):
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    plt.figure(figsize=(7, 6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.title("ROC (one-vs-rest)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Корень с train/ и test/")
    parser.add_argument("--val_dir", type=str, default=None,
                        help="Папка с классами для оценки (single_dir режим)")
    parser.add_argument("--device", type=str, default=CFG.device)
    parser.add_argument("--no_tta", action="store_true",
                        help="Отключить TTA")
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    ckpt = torch.load(args.ckpt, map_location=device)
    class_names = ckpt["class_names"]
    backbone = ckpt["backbone"]
    img_size = ckpt["img_size"]

    cfg = CFG
    cfg.backbone = backbone
    cfg.class_names = tuple(class_names)
    cfg.num_classes = len(class_names)
    cfg.img_size = img_size

    model, _, _ = build_model(backbone, cfg.num_classes, cfg.dropout)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # ---- Определяем, откуда брать val ----
    if args.val_dir:
        # single_dir режим
        val_ds = BrainTumorDataset(args.val_dir, cfg.class_names,
                                   transform=get_val_transforms(cfg.img_size))
    else:
        if args.data_dir:
            cfg.val_dir = Path(args.data_dir) / "test"
        # Только val_loader нужен
        _, val_loader, _, val_ds = build_dataloaders(cfg)

    print(f"[INFO] Val samples: {len(val_ds)}")

    # ---- Инференс ----
    use_tta = not args.no_tta
    if use_tta:
        print("[INFO] TTA enabled")
        probs, labels = run_inference_tta(model, val_ds, device, cfg)
    else:
        # Создаём DataLoader вручную, т.к. val_ds уже с transform
        loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
        probs, labels = run_inference_plain(model, loader, device)

    preds = probs.argmax(axis=1)

    print("\n=== Classification report ===")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    # ---- Recall опухолевых классов ----
    recalls = recall_score(labels, preds, average=None,
                           labels=list(range(cfg.num_classes)), zero_division=0)
    tumor_recall = np.mean([recalls[i] for i in cfg.tumor_class_indices])
    print(f"Recall опухолевых классов (avg по glioma/meningioma/pituitary): "
          f"{tumor_recall:.4f}")

    # ---- Пороговый анализ: сколько опухолей пропущено ----
    # Считаем, сколько истинных опухолей было предсказано как notumor
    tumor_mask = np.isin(labels, list(cfg.tumor_class_indices))
    notumor_idx = cfg.class_names.index("notumor")
    missed = np.sum(tumor_mask & (preds == notumor_idx))
    total_tumors = tumor_mask.sum()
    print(f"Пропущено опухолей (pred=notumor, true=tumor): "
          f"{missed}/{total_tumors} ({missed / max(total_tumors, 1) * 100:.2f}%)")

    # ---- Графики ----
    out_dir = Path("checkpoints")
    plot_confusion(labels, preds, class_names, out_dir / "confusion_matrix.png")
    plot_roc(labels, probs, class_names, out_dir / "roc_curves.png")
    print(f"[DONE] Графики сохранены в {out_dir}/")


if __name__ == "__main__":
    main()