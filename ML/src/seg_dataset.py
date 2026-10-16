"""
Датасет для обучения сегментации.
Добавлен ElasticTransform — главный аугментатор против переобучения на малых данных.
"""
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegmentationDataset(Dataset):
    """Пары (снимок, маска). Аугментации синхронные."""

    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise RuntimeError(f"Не читается: {img_path} / {mask_path}")

        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]
        else:
            img = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
            img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            mask = torch.from_numpy(mask).float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return img, mask


def find_pairs(images_root, masks_root, class_names):
    """Сопоставляет снимки и маски по имени файла (без расширения)."""
    images_root = Path(images_root)
    masks_root = Path(masks_root)

    pairs = []
    class_counts = {}

    for cls in class_names:
        img_dir = images_root / cls
        mask_dir = masks_root / cls

        if not mask_dir.is_dir():
            print(f"[WARN] Нет папки с масками: {mask_dir}")
            continue

        mask_map = {}
        for ext in IMG_EXTS:
            for f in mask_dir.glob(f"*{ext}"):
                mask_map[f.stem] = f

        count = 0
        for ext in IMG_EXTS:
            for img_path in sorted(img_dir.glob(f"*{ext}")):
                if img_path.stem in mask_map:
                    pairs.append((img_path, mask_map[img_path.stem]))
                    count += 1
        class_counts[cls] = count

    return pairs, class_counts


def get_seg_transforms(img_size=256, is_train=True):
    """
    Аугментации для сегментации.
    Включает ElasticTransform — самый эффективный против переобучения.
    """
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),

            # Геометрические
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, border_mode=0, p=0.5),
            A.Affine(
                translate_percent=(-0.08, 0.08),
                scale=(0.85, 1.15),
                rotate=(-20, 20),
                shear=(-8, 8),
                border_mode=0,
                p=0.5,
            ),

            # Эластичные деформации — ключевое улучшение
            A.OneOf([
                A.ElasticTransform(alpha=30, sigma=5, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
                A.OpticalDistortion(distort_limit=0.3, p=1.0),
            ], p=0.5),

            # Яркость / контраст
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),

            # Шум и размытие
            A.OneOf([
                A.GaussNoise(std_range=(0.04, 0.12), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.3),

            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


def get_tta_transforms(img_size=256):
    """TTA-трансформы для инференса (детерминированные)."""
    norm = [
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return [
        A.Compose(norm),
        A.Compose([A.HorizontalFlip(p=1.0)] + norm),
        A.Compose([A.VerticalFlip(p=1.0)] + norm),
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + norm),
    ]