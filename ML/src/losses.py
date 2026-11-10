import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss для мультиклассовой классификации.
    alpha — тензор весов классов (или None).
    gamma — фокусирующий параметр: чем больше, тем сильнее фокус на сложных примерах.
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)  # вероятность правильного класса
        focal = (1.0 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


def build_criterion(cfg, class_weights=None, device="cpu"):
    """
    Возвращает функцию потерь на основе cfg.loss_type.
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)

    if cfg.loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=cfg.focal_gamma)
    elif cfg.loss_type == "ce":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=cfg.label_smoothing,
        )
    else:
        raise ValueError(f"Неизвестный loss_type: {cfg.loss_type}")