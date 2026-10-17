"""
Обёртка над CV-модулем для вызова из FastAPI.
Загружает модель один раз при старте, принимает путь к снимку,
возвращает предсказание и путь к визуализации.
"""
import os
import sys
from pathlib import Path

# Добавляем папку ML в sys.path, чтобы работал импорт src.*
ML_DIR = Path(__file__).parent.parent / "ML"
sys.path.insert(0, str(ML_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ML.src.config import CFG
from ML.src.model import build_model
from ML.src.transforms import get_tta_transforms, get_val_transforms
from ML.src.gradcam_utils import GradCAM, overlay_heatmap

# === Загружаем модель один раз при импорте ===
DEFAULT_CKPT = ML_DIR / "checkpoints" / "best_efficientnet_b0.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_class_names = None
_img_size = None


def _load_model():
    """Ленивая загрузка модели — при первом вызове."""
    global _model, _class_names, _img_size
    if _model is not None:
        return

    if not DEFAULT_CKPT.exists():
        raise FileNotFoundError(
            f"Чекпоинт не найден: {DEFAULT_CKPT}. "
            f"Сначала обучите модель: python -m src.train --single_dir data"
        )

    print(f"[CV] Загружаю модель: {DEFAULT_CKPT}")
    ckpt = torch.load(DEFAULT_CKPT, map_location=DEVICE)
    _class_names = ckpt["class_names"]
    backbone = ckpt["backbone"]
    _img_size = ckpt.get("img_size", 224)

    model, _, _ = build_model(backbone, len(_class_names), CFG.dropout)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    _model = model
    print(f"[CV] Модель загружена. Классы: {_class_names}")


def _find_target_layer(model):
    """Ищем последний Conv2d-слой для Grad-CAM."""
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise RuntimeError("Не найден свёрточный слой")


def predict(image_path: str, use_tta: bool = True) -> dict:
    """
    Основная функция для FastAPI.

    Args:
        image_path: путь к оригинальному снимку (PNG/JPG)
        use_tta: использовать ли TTA

    Returns:
        dict:
            - prediction: класс ("glioma"|"meningioma"|"pituitary"|"notumor")
            - confidence: float (0..1)
            - result_path: путь к сохранённой визуализации (PNG)
            - explanation: текстовое пояснение
    """
    _load_model()

    # 1. Загружаем изображение
    img_rgb = np.array(Image.open(image_path).convert("RGB"))

    # 2. Инференс с TTA или без
    if use_tta:
        tta_transforms = get_tta_transforms(_img_size)
        probs_sum = np.zeros(len(_class_names), dtype=np.float64)
        for tf in tta_transforms:
            tensor = tf(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = _model(tensor)
                probs_sum += F.softmax(logits, dim=1).cpu().numpy()[0]
        probs = probs_sum / len(tta_transforms)
    else:
        tensor = get_val_transforms(_img_size)(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = _model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    pred_class = _class_names[pred_idx]
    confidence = float(probs[pred_idx])

    confidence = (confidence * 100 - 1) / 100
    confidence = int(confidence * 100) / 100

    # 3. Grad-CAM визуализация
    tensor = get_val_transforms(_img_size)(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
    target_layer = _find_target_layer(_model)
    cam = GradCAM(_model, target_layer)
    heatmap, _, _ = cam.generate(tensor, target_class=pred_idx)

    overlay = overlay_heatmap(img_rgb, heatmap, alpha=0.45)

    # 4. Сохраняем визуализацию
    from PIL import Image as PILImage
    result_dir = Path(image_path).parent.parent / "processed"
    result_dir.mkdir(parents=True, exist_ok=True)

    import uuid
    result_name = f"{uuid.uuid4()}.png"
    result_path = result_dir / result_name

    PILImage.fromarray(overlay).save(result_path)

    # 5. Текстовое пояснение
    explanation_map = {
        "glioma": "Обнаружено образование с признаками глиомы",
        "meningioma": "Обнаружено образование с признаками менингиомы",
        "pituitary": "Обнаружено образование в области гипофиза",
        "notumor": "Признаков опухолевого образования не обнаружено",
    }

    return {
        "prediction": pred_class,
        "confidence": confidence,
        "result_path": str(result_path),
        "explanation": explanation_map.get(pred_class, ""),
    }