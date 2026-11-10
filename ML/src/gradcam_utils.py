"""
Grad-CAM визуализация для классификатора опухолей мозга.
Показывает, на какие области снимка "смотрит" модель при принятии решения.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from .transforms import get_val_transforms
from .config import CFG


class GradCAM:
    """
    Простая реализация Grad-CAM без внешних зависимостей.
    Работает для любого nn.Module с CNN-бэкбоном.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        input_tensor: (1, 3, H, W)
        target_class: int или None (= argmax)
        Возвращает: heatmap (H, W) в диапазоне [0, 1] и предсказанный класс.
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)              # (1, num_classes)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # Backward от логита целевого класса
        score = logits[0, target_class]
        score.backward(retain_graph=False)

        # activations: (1, C, h, w); gradients: (1, C, h, w)
        grads = self.gradients                        # dY/dA
        acts = self.activations

        # Веса: среднее по пространственным осям
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Взвешенная сумма активаций + ReLU
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Нормализация в [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.squeeze().cpu().numpy()             # (h, w)

        return cam, target_class, probs.detach().cpu().numpy()[0]


def get_target_layer(model, backbone_name):
    """
    Возвращает последний свёрточный слой для EfficientNet / ResNet / ConvNeXt.
    """
    if backbone_name == "efficientnet_b0":
        # последний блок features
        return model.features[-1]
    elif backbone_name == "resnet50":
        return model.layer4[-1]
    elif backbone_name == "convnext_tiny":
        return model.features[-1]
    else:
        raise ValueError(f"Неизвестный бэкбон: {backbone_name}")


def overlay_heatmap(img_rgb_uint8, cam, alpha=0.45):
    """
    Накладывает heatmap на оригинальное изображение.
    img_rgb_uint8: (H, W, 3) uint8
    cam: (h, w) float в [0,1]
    """
    h, w = img_rgb_uint8.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (img_rgb_uint8.astype(np.float32) * (1 - alpha)
               + heatmap.astype(np.float32) * alpha)
    return overlay.clip(0, 255).astype(np.uint8)


def cam_to_bbox(cam, img_shape, threshold=0.5):
    """
    Пороговая бинаризация heatmap → ограничивающий прямоугольник.
    Возвращает (x1, y1, x2, y2) в координатах оригинального изображения.
    """
    h, w = img_shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    binary = (cam_resized > threshold).astype(np.uint8)

    if binary.sum() == 0:
        return None

    ys, xs = np.where(binary > 0)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return int(x1), int(y1), int(x2), int(y2)


@torch.no_grad()
def _load_image(img_path, img_size):
    img = np.array(Image.open(img_path).convert("RGB"))
    transform = get_val_transforms(img_size)
    tensor = transform(image=img)["image"].unsqueeze(0)
    return img, tensor


def explain_single(model, img_path, class_names, backbone_name, device,
                   target_class=None, save_path=None, show=True):
    """
    Полный пайплайн: загрузка → Grad-CAM → визуализация.
    """
    model.eval()
    model.to(device)

    img_rgb, tensor = _load_image(img_path, CFG.img_size)
    tensor = tensor.to(device)

    target_layer = get_target_layer(model, backbone_name)
    cam = GradCAM(model, target_layer)
    heatmap, pred_class, probs = cam.generate(tensor, target_class=target_class)

    overlay = overlay_heatmap(img_rgb, heatmap, alpha=0.45)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb, cmap="gray")
    axes[0].set_title("Оригинал")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"Grad-CAM (pred: {class_names[pred_class]})")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay — уверенность {probs[pred_class]*100:.1f}%")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    return {
        "predicted_class": class_names[pred_class],
        "probabilities": dict(zip(class_names, probs.tolist())),
        "heatmap": heatmap,
        "overlay": overlay,
    }