import sys
from pathlib import Path

ML_DIR = Path(__file__).parent.parent / "ML"
sys.path.insert(0, str(ML_DIR))

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

from ML.src.model import build_model
from ML.src.transforms import get_val_transforms

DEFAULT_CKPT = ML_DIR / "checkpoints" / "best_efficientnet_orientation.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_class_names = None
_img_size = None


def _load_model():
    global _model, _class_names, _img_size
    if _model is not None:
        return

    if not DEFAULT_CKPT.exists():
        raise FileNotFoundError(
            f"Чекпоинт ориентации не найден: {DEFAULT_CKPT}"
        )

    print(f"[Orientation] Загружаю модель: {DEFAULT_CKPT}")
    ckpt = torch.load(DEFAULT_CKPT, map_location=DEVICE)
    _class_names = ckpt.get("class_names", ("axial", "coronal", "sagittal"))
    backbone = ckpt.get("backbone", "efficientnet_b0")
    _img_size = ckpt.get("img_size", 224)

    model, _, _ = build_model(backbone, len(_class_names), dropout=0.0)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    _model = model
    print(f"[Orientation] Модель загружена. Классы: {_class_names}")


def predict_orientation(image_path: str) -> dict:
    _load_model()

    # 1. Загружаем и обрабатываем картинку
    img = np.array(Image.open(image_path).convert("RGB"))
    transform = get_val_transforms(_img_size)
    tensor = transform(image=img)["image"].unsqueeze(0).to(DEVICE)

    # 2. Инференс
    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    orientation = _class_names[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "orientation": orientation,
        "confidence": confidence
    }