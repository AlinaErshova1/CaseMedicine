import torch.nn as nn
import torchvision.models as models


def build_model(backbone_name: str, num_classes: int, dropout: float = 0.3):
    """
    Возвращает (model, backbone_params_list, head_params_list).
    Параметры — именно list(...), а не generator, чтобы можно было переиспользовать.
    """
    if backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        backbone_params = list(model.features.parameters())
        head_params = list(model.classifier.parameters())

    elif backbone_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2")
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        backbone_params = [p for n, p in model.named_parameters()
                           if not n.startswith("fc.")]
        head_params = list(model.fc.parameters())

    elif backbone_name == "convnext_tiny":
        model = models.convnext_tiny(weights="IMAGENET1K_V1")
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        backbone_params = list(model.features.parameters())
        head_params = list(model.classifier.parameters())

    else:
        raise ValueError(f"Неизвестный бэкбон: {backbone_name}")

    return model, backbone_params, head_params