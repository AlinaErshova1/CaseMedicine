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


class MulticlassSegmentationDataset(Dataset):
    """
    Датасет для многоклассовой сегментации (опухоль, отёк и т.д.).
    Ожидает список пар (путь_к_картинке, путь_к_маске).
    """

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

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]
        else:
            img = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
            img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        return img, mask


def find_multiclass_pairs(images_root, masks_root, class_names):
    """Ищет соответствия между картинками и масками по классам."""
    images_root = Path(images_root)
    masks_root = Path(masks_root)
    pairs = []

    for cls in class_names:
        img_dir = images_root / cls
        mask_dir = masks_root / cls
        if not img_dir.is_dir() or not mask_dir.is_dir():
            continue

        # Собираем маски в словарь для быстрого поиска по имени файла
        mask_map = {}
        for ext in IMG_EXTS:
            for m_path in mask_dir.glob(f"*{ext}"):
                mask_map[m_path.stem] = m_path

        for ext in IMG_EXTS:
            for img_path in sorted(img_dir.glob(f"*{ext}")):
                if img_path.stem in mask_map:
                    pairs.append((img_path, mask_map[img_path.stem]))

    return pairs


def get_seg_transforms(img_size=256, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, border_mode=0, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])