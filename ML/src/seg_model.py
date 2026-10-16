"""
U-Net с энкодером EfficientNet-B0.

Ключевые исправления:
  - Bottleneck = блок 7 (320 каналов, H/32), а НЕ блок 8 (1280)
  - Блок 8 (расширение перед классификатором) пропускается целиком
  - Каналы skip-соединений определяются динамически
  - Параметров: ~5M (было ~15M)
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DecoderBlock(nn.Module):
    """Upsample → concat skip → две свёртки с BN и ReLU."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels,
                               3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class UNetEfficientNet(nn.Module):
    """
    U-Net с энкодером EfficientNet-B0.

    Структура EfficientNet-B0 features (проверено эмпирически):
      [0] out= 32    spatial=H/2       stride 2
      [1] out= 16    spatial=H/2  ← s1
      [2] out= 24    spatial=H/4  ← s2
      [3] out= 40    spatial=H/8  ← s3
      [4] out= 80    spatial=H/16
      [5] out=112    spatial=H/16 ← s4
      [6] out=192    spatial=H/32
      [7] out=320    spatial=H/32 ← BOTTLENECK
      [8] out=1280   spatial=H/32  (расширение, пропускаем)
    """

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.encoder = backbone.features

        # Индексы блоков
        self.skip_block_indices = [1, 2, 3, 5]
        self.bottleneck_block_index = 7      # ← ИСПРАВЛЕНО (было 8)

        # Динамически определяем каналы
        skip_channels, bottleneck_channels = self._probe_encoder()
        print(f"[seg_model] Skip channels: {skip_channels}")
        print(f"[seg_model] Bottleneck channels: {bottleneck_channels}")

        # Декодер
        dec_channels = [256, 128, 64, 32]
        s1_ch = skip_channels[1]
        s2_ch = skip_channels[2]
        s3_ch = skip_channels[3]
        s4_ch = skip_channels[5]

        self.dec4 = DecoderBlock(bottleneck_channels, s4_ch, dec_channels[0])
        self.dec3 = DecoderBlock(dec_channels[0], s3_ch, dec_channels[1])
        self.dec2 = DecoderBlock(dec_channels[1], s2_ch, dec_channels[2])
        self.dec1 = DecoderBlock(dec_channels[2], s1_ch, dec_channels[3])
        self.dec0 = DecoderBlock(dec_channels[3], 0, 16)

        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

        # Инициализация декодера (энкодер уже предобучен)
        self._init_decoder()

    def _probe_encoder(self):
        """Прогон dummy для определения реальных каналов."""
        was_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, 256, 256)
            channels = {}
            for i, block in enumerate(self.encoder):
                x = block(x)
                channels[i] = x.shape[1]
        if was_training:
            self.encoder.train()
        return channels, channels[self.bottleneck_block_index]

    def _init_decoder(self):
        """Kaiming-инициализация только для декодера."""
        for m in [self.dec4, self.dec3, self.dec2, self.dec1, self.dec0]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight,
                                            mode="fan_out",
                                            nonlinearity="relu")
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
        # Head — Xavier
        nn.init.xavier_normal_(self.head.weight)

    def forward(self, x):
        skips = {}
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i in self.skip_block_indices:
                skips[i] = x
            if i == self.bottleneck_block_index:
                break   # блок 8 не выполняем

        x = self.dec4(x, skips[5])
        x = self.dec3(x, skips[3])
        x = self.dec2(x, skips[2])
        x = self.dec1(x, skips[1])
        x = self.dec0(x, None)
        x = self.head(x)
        return x

    def freeze_encoder(self, freeze=True):
        """Замораживает/размораживает энкодер."""
        for p in self.encoder.parameters():
            p.requires_grad = not freeze

    def unfreeze_encoder_bn(self):
        """Возвращает BatchNorm энкодера в train-режим после заморозки."""
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()


def build_unet(pretrained=True, classifier_ckpt=None):
    """Собирает U-Net с опциональной инициализацией энкодера весами классификатора."""
    model = UNetEfficientNet(num_classes=1, pretrained=pretrained)

    if classifier_ckpt is not None:
        ckpt_path = Path(classifier_ckpt)
        if ckpt_path.exists():
            print(f"[INFO] Загружаю веса энкодера из {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt["model_state"]

            encoder_state = {}
            for k, v in state.items():
                if k.startswith("features."):
                    encoder_state[k.replace("features.", "encoder.", 1)] = v

            model_state = model.state_dict()
            loaded = {}
            skipped = []
            for k, v in encoder_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    loaded[k] = v
                else:
                    skipped.append(k)

            model.load_state_dict(loaded, strict=False)
            print(f"[INFO] Загружено слоёв энкодера: {len(loaded)}/"
                  f"{len(encoder_state)} (пропущено {len(skipped)})")
        else:
            print(f"[WARN] Не найден чекпоинт: {ckpt_path}")

    return model
# ============================================================
# LEGACY-версия (bottleneck=1280) — для загрузки старых чекпоинтов
# ============================================================

class UNetEfficientNetLegacy(nn.Module):
    """
    Первая версия U-Net со сломанным bottleneck (1280 вместо 320).
    Нужна ТОЛЬКО для загрузки старых чекпоинтов с суффиксом v1_old.
    Использовать для новых обучений нельзя.
    """

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.encoder = backbone.features

        self.skip_block_indices = [1, 2, 3, 5]
        self.bottleneck_block_index = 8   # ← СТАРОЕ значение

        # Каналы в старой архитектуре
        self.dec4 = DecoderBlock(1280, 112, 256)
        self.dec3 = DecoderBlock(256, 40, 128)
        self.dec2 = DecoderBlock(128, 24, 64)
        self.dec1 = DecoderBlock(64, 16, 32)
        self.dec0 = DecoderBlock(32, 0, 16)
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        skips = {}
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i in self.skip_block_indices:
                skips[i] = x
        x = self.dec4(x, skips[5])
        x = self.dec3(x, skips[3])
        x = self.dec2(x, skips[2])
        x = self.dec1(x, skips[1])
        x = self.dec0(x, None)
        x = self.head(x)
        return x


def build_unet_from_ckpt(ckpt, device="cpu"):
    """
    Собирает U-Net по информации из чекпоинта.
    Автоматически выбирает правильную архитектуру (v1_legacy или v2).
    """
    # Определяем версию архитектуры
    arch = ckpt.get("architecture", "v2")   # по умолчанию v2

    if arch == "v1_legacy":
        print("[seg_model] Использую LEGACY архитектуру (bottleneck=1280)")
        model = UNetEfficientNetLegacy(num_classes=1, pretrained=False)
    else:
        print("[seg_model] Использую v2 архитектуру (bottleneck=320)")
        model = UNetEfficientNet(num_classes=1, pretrained=False)

    # Загружаем веса
    try:
        model.load_state_dict(ckpt["model_state"])
        print("[seg_model] Веса загружены успешно")
    except RuntimeError as e:
        # Fallback: если ключа architecture нет, попробуем угадать по форме тензора
        print(f"[seg_model] Загрузка не удалась, пробую fallback...")
        state = ckpt["model_state"]
        # dec4.conv1.weight имеет форму [256, X, 3, 3], где X — bottleneck+112
        dec4_key = "dec4.conv1.weight"
        if dec4_key in state:
            in_ch = state[dec4_key].shape[1]   # 1392 (legacy) или 432 (v2)
            if in_ch > 1000:
                print(f"[seg_model] По форме тензора ({in_ch} каналов) — LEGACY")
                model = UNetEfficientNetLegacy(num_classes=1, pretrained=False)
            else:
                print(f"[seg_model] По форме тензора ({in_ch} каналов) — v2")
                model = UNetEfficientNet(num_classes=1, pretrained=False)
            model.load_state_dict(state)
        else:
            raise

    model.to(device).eval()
    return model

if __name__ == "__main__":
    model = build_unet(pretrained=False)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    n = sum(p.numel() for p in model.parameters())
    print(f"Параметров: {n / 1e6:.2f}M")