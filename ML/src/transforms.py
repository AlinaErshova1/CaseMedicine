import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int = 224):
    """
    Умеренные аугментации: НЕ используем CoarseDropout и сильный GaussNoise,
    чтобы не закрывать/размывать опухолевую область.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=15,
            border_mode=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5,
        ),
        # Лёгкий шум — на новых версиях параметр std_range вместо var_limit
        A.GaussNoise(std_range=(0.04, 0.15), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size: int = 224):
    """
    Список TTA-трансформов (без случайности — детерминированные версии).
    Каждый даёт чуть другой вход, усредняем softmax.
    """
    base_norm = [
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return [
        # 0: без изменений
        A.Compose(base_norm),
        # 1: горизонтальный флип
        A.Compose([A.HorizontalFlip(p=1.0)] + base_norm),
        # 2: небольшой поворот
        A.Compose([A.Rotate(limit=(10, 10), p=1.0, border_mode=0)] + base_norm),
        # 3: поворот -10 + флип
        A.Compose([A.HorizontalFlip(p=1.0), A.Rotate(limit=(-10, -10), p=1.0, border_mode=0)] + base_norm),
    ]