from pathlib import Path
from pipeline import analyze_mri_pipeline


def format_medical_report(results: dict, patient_code: str = "P-XXXX") -> str:
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

Пациента: {patient_code}
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
"""
    return report

# d
def test_local_pipeline():
    test_image_path = "D:/PycharmProjects/CaseMedicine/ML/data/test_clean/notumor/45 no.jpg"

    image_path = Path(test_image_path)
    if not image_path.exists():
        print(f"[Ошибка] Тестовый файл не найден: {image_path}")
        print("Пожалуйста, проверьте путь к файлу.")
        return

    print(f"=== Запуск локального теста пайплайна для файла: {image_path} ===")

    results_dir = Path("checkpoints/test_output")

    try:
        result = analyze_mri_pipeline(str(image_path), results_dir)

        # 2. Форматируем результат в официальное медицинское заключение
        official_report = format_medical_report(result, patient_code="P-2026-AB12")

        # 3. Выводим результат в консоль
        print(official_report)
        print(" Тест успешно завершен! Все модели отработали корректно.")

    except Exception as e:
        print(f"\n[❌ Ошибка во время выполнения пайплайна]: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_local_pipeline()