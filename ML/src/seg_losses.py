"""
Функции потерь для сегментации.

- DiceBCELoss: стандарт, хорошо работает на сбалансированных данных
- FocalDiceLoss: фокус на сложных примерах
- TverskyLoss: позволяет управлять балансом recall/precision
  (для медицины: beta > alpha → фокус на recall, меньше пропусков)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def tversky_loss(logits, targets, alpha=0.3, beta=0.7, eps=1e-6):
    """
    Tversky Loss. alpha + beta = 1 обычно.
    alpha — вес FP, beta — вес FN.
    beta > alpha → модель штрафуется за пропуски (FN) сильнее → выше recall.
    Рекомендация для медицины: alpha=0.3, beta=0.7.
    """
    probs = torch.sigmoid(logits)
    tp = (probs * targets).sum(dim=(2, 3))
    fp = (probs * (1 - targets)).sum(dim=(2, 3))
    fn = ((1 - probs) * targets).sum(dim=(2, 3))
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1 - tversky.mean()


class DiceBCELoss(nn.Module):
    """Классическая связка: Dice + BCE (равные веса)."""

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = dice_loss(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


class FocalDiceLoss(nn.Module):
    """Focal + Dice для несбалансированных данных."""

    def __init__(self, alpha=0.25, gamma=2.0, focal_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = (self.alpha * (1 - pt) ** self.gamma * bce).mean()
        dice = dice_loss(logits, targets)
        return self.focal_weight * focal + (1 - self.focal_weight) * dice


class TverskyBCELoss(nn.Module):
    """
    Tversky + BCE с фокусом на recall.
    Рекомендуется для задач, где пропуск опухоли хуже, чем ложная тревога.
    """

    def __init__(self, alpha=0.3, beta=0.7, bce_weight=0.4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        tv = tversky_loss(logits, targets, self.alpha, self.beta)
        return self.bce_weight * bce + (1 - self.bce_weight) * tv