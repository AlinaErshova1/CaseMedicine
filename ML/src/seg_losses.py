import torch
import torch.nn as nn
import torch.nn.functional as F

class MulticlassDiceBCELoss(nn.Module):
    """Комбинация CrossEntropy и многоклассового Dice Loss для ядра и отёка."""
    def __init__(self, num_classes=3, bce_weight=0.4, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        # Стабильный CrossEntropy по пикселям
        ce_loss = F.cross_entropy(logits, targets)
        
        # Перевод в вероятности (Softmax)
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        dice_loss_val = 0.0
        for c in range(self.num_classes):
            p_c = probs[:, c, :, :]
            t_c = targets_one_hot[:, c, :, :]
            
            intersection = (p_c * t_c).sum(dim=(1, 2))
            union = p_c.sum(dim=(1, 2)) + t_c.sum(dim=(1, 2))
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss_val += (1.0 - dice.mean())
            
        dice_loss_val /= self.num_classes
        return self.bce_weight * ce_loss + (1.0 - self.bce_weight) * dice_loss_val