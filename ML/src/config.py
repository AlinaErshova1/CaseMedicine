from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Пути
    data_dir: Path = Path("data")
    train_dir: Path = Path("data/train")
    val_dir: Path = Path("data/test")
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")

    # Классы (порядок важен — соответствует индексам 0..3)
    class_names: tuple = ("glioma", "meningioma", "notumor", "pituitary")
    num_classes: int = 4
    # Индексы классов, которые считаются "опухоль" — для расчёта recall_tumor
    tumor_class_indices: tuple = (0, 1, 3)  # всё кроме notumor (idx=2)

    # Изображения
    img_size: int = 224
    in_channels: int = 3

    # Обучение
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 25
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    dropout: float = 0.3

    # Loss
    loss_type: str = "focal"    # "ce" | "focal"
    focal_gamma: float = 2.0

    # Метрика для сохранения лучшей модели
    monitor_metric: str = "recall_tumor"   # "recall_tumor" | "f1_macro" | "acc"

    # Модель
    backbone: str = "efficientnet_b0"

    # TTA
    use_tta: bool = True
    tta_n: int = 4   # количество аугментаций

    # Прочее
    seed: int = 42
    device: str = "cuda"
    use_class_weights: bool = True
    early_stopping_patience: int = 7

    def __post_init__(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


CFG = Config()