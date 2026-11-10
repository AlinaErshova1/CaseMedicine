import argparse
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .seg_dataset import MulticlassSegmentationDataset, find_multiclass_pairs, get_seg_transforms
from .seg_model import build_multiclass_unet
from .seg_losses import MulticlassDiceBCELoss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="data/train")
    parser.add_argument("--masks_dir", type=str, default="data/masks/train")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    class_names = ["glioma", "meningioma", "pituitary"]

    pairs = find_multiclass_pairs(args.images_dir, args.masks_dir, class_names)
    print(f"[INFO] Найдено пар изображение-маска: {len(pairs)}")
    if len(pairs) == 0:
        print("[ERROR] Пары не найдены! Проверьте папки data/train и data/masks/train")
        return

    train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

    train_ds = MulticlassSegmentationDataset(train_pairs, transform=get_seg_transforms(256, is_train=True))
    val_ds = MulticlassSegmentationDataset(val_pairs, transform=get_seg_transforms(256, is_train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 3 класса: 0-фон, 1-ядро, 2-отёк
    model = build_multiclass_unet(num_classes=3, pretrained=True).to(device)
    criterion = MulticlassDiceBCELoss(num_classes=3)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    Path("checkpoints").mkdir(exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs) # (B, 3, H, W)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "checkpoints/best_multiclass_unet.pth")
            print("  ↳ Модель сохранена!")

if __name__ == "__main__":
    main()