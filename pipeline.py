import os
import sys
from pathlib import Path
import uuid
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

# Добавляем путь к папке ML
ML_DIR = Path(__file__).parent / "ML"
sys.path.insert(0, str(ML_DIR))

from ML.src.config import CFG
from ML.src.model import build_model
from ML.src.transforms import get_val_transforms, get_tta_transforms
from ML.src.gradcam_utils import GradCAM, overlay_heatmap
from ML.src.seg_dataset import IMAGENET_MEAN, IMAGENET_STD
from ML.src.seg_model import build_unet_from_ckpt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ORIENT_CKPT = ML_DIR / "checkpoints" / "best_efficientnet_orientation.pth"
TUMOR_CKPT = ML_DIR / "checkpoints" / "best_efficientnet_b0.pth"
UNET_CKPT = ML_DIR / "checkpoints" / "best_unet.pth"

_orient_model = None
_orient_meta = None

_tumor_model = None
_tumor_classes = None
_tumor_img_size = 224

_unet_model = None
_unet_size = 256
_unet_threshold = 0.75


def load_all_models():
    global _orient_model, _orient_meta, _tumor_model, _tumor_classes, _tumor_img_size, _unet_model, _unet_size, _unet_threshold

    # 1. Модель ориентации
    if _orient_model is None:
        if ORIENT_CKPT.exists():
            print(f"[Pipeline] Загрузка модели ориентации: {ORIENT_CKPT}")
            ckpt = torch.load(ORIENT_CKPT, map_location=DEVICE)
            _orient_meta = ckpt.get("class_names", ("axial", "coronal", "sagittal"))
            backbone = ckpt.get("backbone", "efficientnet_b0")
            model, _, _ = build_model(backbone, len(_orient_meta), dropout=0.0)
            model.load_state_dict(ckpt["model_state"])
            model.to(DEVICE).eval()
            _orient_model = model
        else:
            print(f"[Pipeline Warning] Чекпоинт ориентации не найден: {ORIENT_CKPT}")

    # 2. Классификатор опухолей
    if _tumor_model is None:
        if TUMOR_CKPT.exists():
            print(f"[Pipeline] Загрузка классификатора опухолей: {TUMOR_CKPT}")
            ckpt = torch.load(TUMOR_CKPT, map_location=DEVICE)
            _tumor_classes = ckpt["class_names"]
            backbone = ckpt["backbone"]
            _tumor_img_size = ckpt.get("img_size", 224)
            model, _, _ = build_model(backbone, len(_tumor_classes), CFG.dropout)
            model.load_state_dict(ckpt["model_state"])
            model.to(DEVICE).eval()
            _tumor_model = model
        else:
            print(f"[Pipeline Warning] Чекпоинт классификатора не найден: {TUMOR_CKPT}")

    # 3. U-Net сегментация
    if _unet_model is None:
        if UNET_CKPT.exists():
            print(f"[Pipeline] Загрузка U-Net: {UNET_CKPT}")
            ckpt = torch.load(UNET_CKPT, map_location=DEVICE)
            _unet_size = ckpt.get("img_size", 256)
            _unet_threshold = ckpt.get("threshold", 0.75)

            model = build_unet_from_ckpt(ckpt, device=DEVICE)
            _unet_model = model
        else:
            print(f"[Pipeline Warning] Чекпоинт U-Net не найден: {UNET_CKPT}")


def analyze_mri_pipeline(image_path: str, results_dir: Path) -> dict:
    """
    Строгий медицинский пайплайн анализа МРТ:
    Шаг 1 -> Определение ориентации
    Шаг 2 -> Классификация типа опухоли
    Шаг 3 -> Сегментация (U-Net) + Анатомическая эвристика локализации
    """
    load_all_models()
    img_rgb = np.array(Image.open(image_path).convert("RGB"))
    orig_h, orig_w = img_rgb.shape[:2]

    results = {
        "orientation": "axial",
        "orientation_confidence": 0.0,
        "prediction": "notumor",
        "confidence": 0.0,
        "has_tumor": False,
        "area_pct": 0.0,
        "localization": "Не определено",
        "result_path": None,
        "mask_path": None,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    file_uid = str(uuid.uuid4())

    # ==========================================
    # ШАГ 1: ОПРЕДЕЛЕНИЕ ОРИЕНТАЦИИ СРЕЗА
    # ==========================================
    if _orient_model is not None:
        transform = get_val_transforms(224)
        tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = _orient_model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        results["orientation"] = _orient_meta[pred_idx]
        results["orientation_confidence"] = float(probs[pred_idx])

    # ==========================================
    # ШАГ 2: КЛАССИФИКАЦИЯ ОПУХОЛИ (EfficientNet)
    # ==========================================
    if _tumor_model is not None:
        transform = get_val_transforms(_tumor_img_size)
        tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = _tumor_model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        results["prediction"] = _tumor_classes[pred_idx]
        results["confidence"] = float(probs[pred_idx])

        # Grad-CAM визуализация для классификатора
        try:
            target_layer = [m for m in reversed(list(_tumor_model.modules())) if isinstance(m, torch.nn.Conv2d)][0]
            cam = GradCAM(_tumor_model, target_layer)
            heatmap, _, _ = cam.generate(tensor, target_class=pred_idx)
            overlay = overlay_heatmap(img_rgb, heatmap, alpha=0.45)

            cls_res_path = results_dir / f"cls_{file_uid}.png"
            Image.fromarray(overlay).save(cls_res_path)
            results["result_path"] = str(cls_res_path)
        except Exception as e:
            print(f"[Pipeline Warning] Ошибка генерации Grad-CAM: {e}")

    # ==========================================
    # ШАГ 3: СЕГМЕНТАЦИЯ (U-Net) И ЭВРИСТИКА ЛОКАЛИЗАЦИИ
    # ==========================================
    if _unet_model is not None and results["prediction"] != "notumor":
        try:
            # Предопроцессинг для U-Net
            x_u = cv2.resize(img_rgb, (_unet_size, _unet_size)).astype(np.float32) / 255.0
            x_u = (x_u - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
            tensor_u = torch.from_numpy(x_u.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

            with torch.no_grad():
                logits_u = _unet_model(tensor_u)
                probs_u = torch.sigmoid(logits_u)[0, 0].cpu().numpy()

            # Ресайз маски к оригинальному размеру снимка
            probs_resized = cv2.resize(probs_u, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            mask_binary = (probs_resized > _unet_threshold).astype(np.uint8)

            area_px = int(mask_binary.sum())
            area_pct = round(area_px / mask_binary.size * 100, 2)
            has_tumor = area_pct > 0.3

            results["has_tumor"] = has_tumor
            results["area_pct"] = area_pct

            # Накладываем красную маску на оригинал для красоты
            overlay = img_rgb.copy()
            overlay[mask_binary > 0] = [255, 60, 60]  # Красный цвет
            blended = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)

            # Сохраняем картинку с выделением
            final_result_path = results_dir / f"unet_overlay_{file_uid}.png"
            cv2.imwrite(str(final_result_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            results["result_path"] = str(final_result_path)

            # --- АНАТОМИЧЕСКАЯ ЭВРИСТИКА С УЧЕТОМ ТИПА ПЛОСКОСТИ ---
            y_idx, x_idx = np.where(mask_binary > 0)
            if len(x_idx) > 0:
                center_x = np.mean(x_idx)
                center_y = np.mean(y_idx)

                orientation = results.get("orientation", "axial").lower()
                midline_x = orig_w / 2

                hemi_str = "не определяется в данной плоскости"
                dept_str = "не определяется в данной плоскости"

                if orientation == "axial":
                    hemi_str = "правое полушарие" if center_x < midline_x else "левое полушарие"
                    if center_y < orig_h * 0.4:
                        dept_str = "лобная доля / передние отделы"
                    elif center_y > orig_h * 0.7:
                        dept_str = "затылочная доля / задние отделы"
                    else:
                        dept_str = "теменная/височная доля / центральные отделы"

                elif orientation == "coronal":
                    hemi_str = "правое полушарие" if center_x < midline_x else "левое полушарие"
                    dept_str = "не определяется в корональной плоскости"

                elif orientation == "sagittal":
                    hemi_str = "не определяется в сагиттальной плоскости"
                    if center_x < midline_x:
                        dept_str = "передние отделы (лобная зона)"
                    else:
                        dept_str = "задние отделы (затылочная/теменная зона)"

                results["localization"] = f"Полушарие: {hemi_str}, Отдел: {dept_str}"

        except Exception as e:
            print(f"[Pipeline Warning] Ошибка U-Net / эвристики: {e}")

    return results


def format_medical_report(results: dict, patient_code: str = "P-XXXX") -> str:
    """
    Форматирует результаты пайплайна в официальный медицинский протокол.
    """
    orientation_map = {
        "axial": "аксиальная",
        "coronal": "корональная",
        "sagittal": "сагиттальная"
    }

    prediction_map = {
        "glioma": "МР-картина, наиболее соответствующая глиальной опухоли (глиома)",
        "meningioma": "МР-картина, наиболее соответствующая менингиоме",
        "pituitary": "МР-картина образования области турецкого седла (аденома гипофиза)",
        "notumor": "Данных за объемное образование головного мозга не выявлено"
    }

    orient_ru = orientation_map.get(results.get("orientation", "axial"), "аксиальная")
    pred_key = results.get("prediction", "notumor")
    conclusion_text = prediction_map.get(pred_key, "Патологические изменения")
    confidence = results.get("confidence", 0.0) * 100
    has_tumor = results.get("has_tumor", False)
    area_pct = results.get("area_pct", 0.0)
    localization = results.get("localization", "не определена")

    report = f"""
====================================================================
               ПРОТОКОЛ ЛУЧЕВОЙ ДИАГНОСТИКИ (ИИ-АНАЛИЗ МРТ)
====================================================================
Идентификатор пациента: {patient_code}
Плоскость сканирования: {orient_ru} (уверенность модели: {results.get('orientation_confidence', 0) * 100:.1f}%)

1. ОПИСАТЕЛЬНАЯ ЧАСТЬ:
При анализе представленного МРТ-изображения головного мозга проведена автоматизированная сегментация и классификация паттернов ткани. 
- Наличие патологического очага: {"выявлен" if has_tumor else "не выявлен"}
- Относительный объем / площадь поражения: {area_pct}% от площади среза.
- Анатомическая локализация: {localization}.

2. ЗАКЛЮЧЕНИЕ:
{conclusion_text}. 
Вероятность (уверенность алгоритма СППР): {confidence:.1f}%.

3. РЕКОМЕНДАЦИИ:
- Настоящий протокол получен с использованием системы поддержки принятия врачебных решений (СППР) и не является окончательным клиническим диагнозом.
- Рекомендована очная консультация врача-невролога / нейрохирурга и сопоставление с полным пакетом исходных DICOM-данных исследования.

Дата формирования протокола: автоматизированно
Подпись ответственного специалиста: _________________ / Врач-модератор
====================================================================
"""
    return report
