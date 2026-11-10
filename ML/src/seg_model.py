import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x

class UNetEfficientNetMulticlass(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.encoder = backbone.features

        self.skip_block_indices = [1, 2, 3, 5]
        self.bottleneck_block_index = 7

        _, bottleneck_channels = self._probe_encoder()

        dec_channels = [256, 128, 64, 32]
        self.dec4 = DecoderBlock(bottleneck_channels, 112, dec_channels[0])
        self.dec3 = DecoderBlock(dec_channels[0], 40, dec_channels[1])
        self.dec2 = DecoderBlock(dec_channels[1], 24, dec_channels[2])
        self.dec1 = DecoderBlock(dec_channels[2], 16, dec_channels[3])
        self.dec0 = DecoderBlock(dec_channels[3], 0, 16)

        # Выходной слой на num_classes каналов (фон, ядро, отёк)
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def _probe_encoder(self):
        self.encoder.eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, 256, 256)
            channels = {}
            for i, block in enumerate(self.encoder):
                x = block(x)
                channels[i] = x.shape[1]
        return channels, channels[self.bottleneck_block_index]

    def forward(self, x):
        skips = {}
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i in self.skip_block_indices:
                skips[i] = x
            if i == self.bottleneck_block_index:
                break

        x = self.dec4(x, skips[5])
        x = self.dec3(x, skips[3])
        x = self.dec2(x, skips[2])
        x = self.dec1(x, skips[1])
        x = self.dec0(x, None)
        return self.head(x)

def build_multiclass_unet(num_classes=3, pretrained=True):
    return UNetEfficientNetMulticlass(num_classes=num_classes, pretrained=pretrained)