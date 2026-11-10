from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class BrainTumorDataset(Dataset):
    """
    Ожидает структуру:
        root/
          class_1/ *.png
          class_2/ *.png
          ...
    """

    def __init__(self, root_dir, class_names, transform=None, samples=None):
        self.root_dir = Path(root_dir)
        self.class_names = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.transform = transform

        if samples is None:
            samples = self._scan()
        # Сортируем для воспроизводимости
        self.samples = sorted(samples, key=lambda x: x[0])

    def _scan(self):
        samples = []
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Нет папки: {class_dir}")
            for ext in IMG_EXTS:
                for f in sorted(class_dir.glob(f"*{ext}")):
                    samples.append((str(f), self.class_to_idx[class_name]))
        if len(samples) == 0:
            raise RuntimeError(f"Не найдено изображений в {self.root_dir}")
        return samples

    def class_counts(self):
        counts = {c: 0 for c in self.class_names}
        for _, y in self.samples:
            counts[self.class_names[y]] += 1
        return counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = np.array(img)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img, label


def stratified_split(root_dir, class_names, val_ratio=0.2, seed=42):
    """Стратифицированный split train/val из одной папки."""
    root_dir = Path(root_dir)
    all_samples = []
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    for class_name in class_names:
        class_dir = root_dir / class_name
        for ext in IMG_EXTS:
            for f in sorted(class_dir.glob(f"*{ext}")):
                all_samples.append((str(f), class_to_idx[class_name]))

    paths = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    train_p, val_p, train_y, val_y = train_test_split(
        paths, labels, test_size=val_ratio, stratify=labels, random_state=seed
    )
    return list(zip(train_p, train_y)), list(zip(val_p, val_y))


def compute_class_weights(samples, num_classes):
    """Веса обратно пропорциональны частоте класса."""
    labels = np.array([y for _, y in samples])
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(cfg, train_samples=None, val_samples=None,
                      train_dir=None, val_dir=None):
    """
    Если train_samples/val_samples переданы — используем их (single_dir режим).
    Иначе читаем train_dir и val_dir (или cfg.train_dir / cfg.val_dir).
    """
    from .transforms import get_train_transforms, get_val_transforms

    train_dir = Path(train_dir) if train_dir else cfg.train_dir
    val_dir = Path(val_dir) if val_dir else cfg.val_dir

    if train_samples is None:
        train_ds = BrainTumorDataset(train_dir, cfg.class_names,
                                     transform=get_train_transforms(cfg.img_size))
    else:
        train_ds = BrainTumorDataset(train_dir, cfg.class_names,
                                     transform=get_train_transforms(cfg.img_size),
                                     samples=train_samples)

    if val_samples is None:
        val_ds = BrainTumorDataset(val_dir, cfg.class_names,
                                   transform=get_val_transforms(cfg.img_size))
    else:
        val_ds = BrainTumorDataset(train_dir, cfg.class_names,
                                   transform=get_val_transforms(cfg.img_size),
                                   samples=val_samples)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin,
    )
    return train_loader, val_loader, train_ds, val_ds