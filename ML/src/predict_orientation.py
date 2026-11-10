import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


from .model import build_model
from .transforms import get_val_transforms


@torch.no_grad()
def predict_slice_orientation(img_path, ckpt_path, device="cpu"):
    """
    Определяет проекцию среза МРТ: axial, coronal или sagittal.
    """
    device = torch.device(device)

    # Загружаем чекпоинт обученного классификатора ориентации
    ckpt = torch.load(ckpt_path, map_location=device)
    class_names = ckpt.get("class_names", ("axial", "coronal", "sagittal"))
    backbone = ckpt.get("backbone", "efficientnet_b0")
    img_size = ckpt.get("img_size", 224)

    # Собираем модель под 3 класса
    model, _, _ = build_model(backbone, len(class_names), dropout=0.0)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # Предобработка изображения
    img = np.array(Image.open(img_path).convert("RGB"))
    transform = get_val_transforms(img_size)
    tensor = transform(image=img)["image"].unsqueeze(0).to(device)

    # Инференс
    logits = model(tensor)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))

    orientation = class_names[pred_idx]
    confidence = probs[pred_idx] * 100

    print(f"[INFO] Срез: {img_path}")
    print(f"  → Определенная проекция: {orientation.upper()} ({confidence:.2f}% уверенности)")

    return orientation, probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Путь к МРТ-снимку")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_efficientnet_orientation.pth")
    args = parser.parse_args()

    predict_slice_orientation(args.image, args.ckpt)