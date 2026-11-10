"""
Обёртка над U-Net для сегментации опухоли.
Загружает модель один раз, возвращает маску и наложенное изображение.
"""
import os
import sys
from pathlib import Path
import uuid

# ML в sys.path
ML_DIR = Path(__file__).parent.parent / "ML"
sys.path.insert(0, str(ML_DIR))

import numpy as np
import torch
import cv2
from PIL import Image

from ML.src.seg_model import build_unet_from_ckpt
from ML.src.seg_dataset import IMAGENET_MEAN, IMAGENET_STD


# === Загрузка модели ===
DEFAULT_CKPT = ML_DIR / "checkpoints" / "best_unet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_img_size = None
_threshold = 0.75   # оптимальный порог из метрик


def _load_model():
    """Ленивая загрузка U-Net."""
    global _model, _img_size, _threshold
    if _model is not None:
        return

    if not DEFAULT_CKPT.exists():
        raise FileNotFoundError(
            f"Чекпоинт U-Net не найден: {DEFAULT_CKPT}"
        )

    print(f"[UNet] Загружаю модель: {DEFAULT_CKPT}")
    ckpt = torch.load(DEFAULT_CKPT, map_location=DEVICE)
    _img_size = ckpt.get("img_size", 256)
    _threshold = ckpt.get("threshold", 0.75)  # из метаданных, если есть

    _model = build_unet_from_ckpt(ckpt, device=DEVICE)
    print(f"[UNet] Модель загружена. img_size={_img_size}, threshold={_threshold}")


def _preprocess(img_rgb: np.ndarray, img_size: int) -> torch.Tensor:
    """Ресайз + нормализация ImageNet."""
    x = cv2.resize(img_rgb, (img_size, img_size)).astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    return x


def _overlay_mask(image_rgb: np.ndarray, mask_binary: np.ndarray,
                  color=(255, 60, 60), alpha=0.45) -> np.ndarray:
    """Накладывает маску поверх снимка."""
    overlay = image_rgb.copy()
    overlay[mask_binary > 0] = color
    blended = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)
    return blended


def predict_mask(image_path: str) -> dict:
    """
    Основная функция для FastAPI.

    Args:
        image_path: путь к оригинальному снимку

    Returns:
        dict:
            - mask_path: путь к сохранённой маске (PNG, чёрно-белая)
            - overlay_path: путь к наложенному изображению (PNG)
            - area_pct: процент площади опухоли от всего снимка
            - has_tumor: bool — есть ли опухоль (по площади)
    """
    _load_model()

    # 1. Загружаем снимок
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Не читается: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # 2. Инференс
    with torch.no_grad():
        x = _preprocess(img_rgb, _img_size)
        logits = _model(x)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # 3. Ресайз маски к оригинальному размеру
    probs_resized = cv2.resize(probs, (orig_w, orig_h),
                                interpolation=cv2.INTER_LINEAR)

    # 4. Бинаризация по порогу
    mask_binary = (probs_resized > _threshold).astype(np.uint8)

    # 5. Площадь опухоли
    area_px = int(mask_binary.sum())
    area_pct = area_px / mask_binary.size * 100
    has_tumor = area_pct > 0.5   # если больше 0.5% площади — считаем, что опухоль есть

    # 6. Сохраняем результат
    result_dir = Path(__file__).parent / "uploads" / "processed"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Маска (чёрно-белая)
    mask_name = f"unet_mask_{uuid.uuid4()}.png"
    mask_path = result_dir / mask_name
    cv2.imwrite(str(mask_path), (mask_binary * 255).astype(np.uint8))

    # Наложенное изображение
    overlay_rgb = _overlay_mask(img_rgb, mask_binary)
    overlay_name = f"unet_overlay_{uuid.uuid4()}.png"
    overlay_path = result_dir / overlay_name
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    return {
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "area_pct": round(area_pct, 2),
        "has_tumor": has_tumor,
    }