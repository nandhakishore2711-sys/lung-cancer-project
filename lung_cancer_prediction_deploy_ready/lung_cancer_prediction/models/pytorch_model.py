"""
PyTorch model definitions for lung cancer CT scan classification.

Mirrors models/tensorflow_model.py:

1. LungCancerCNN     - the custom "novel" architecture, four conv blocks.
2. build_transfer_model - EfficientNet-B0 backbone (torchvision) for a
                           transfer-learning comparison baseline.

Both are exposed through build_model(architecture=...) so app.py and
train_pytorch.py share one entry point, just like the TF side.
"""

import torch
import torch.nn as nn
import torchvision


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class LungCancerCNN(nn.Module):
    """Custom CNN — same topology as the Keras build_custom_cnn."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.block1 = ConvBlock(3, 32, dropout=0.25)
        self.block2 = ConvBlock(32, 64, dropout=0.25)
        self.block3 = ConvBlock(64, 128, dropout=0.3)

        # Block 4 has no pooling here so the Grad-CAM feature map stays a
        # reasonable spatial resolution (matches the TF model's last_conv).
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.last_conv_channels = 256

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

        # Kept for Grad-CAM hooking without re-walking the module tree.
        self._last_conv_layer = self.block4[3]  # second Conv2d in block4

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        features = x  # (B, 256, H, W) — Grad-CAM taps this
        x = self.gap(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits, features

    def get_last_conv_layer(self):
        return self._last_conv_layer


def build_transfer_model(num_classes=3, fine_tune=False):
    """EfficientNet-B0 backbone via torchvision, ImageNet-pretrained."""
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    base = torchvision.models.efficientnet_b0(weights=weights)

    for param in base.features.parameters():
        param.requires_grad = fine_tune

    in_features = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
    return base


def build_model(architecture="custom", num_classes=3):
    if architecture == "custom":
        return LungCancerCNN(num_classes=num_classes)
    elif architecture == "efficientnet":
        return build_transfer_model(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


if __name__ == "__main__":
    model = build_model("custom")
    dummy = torch.randn(2, 3, 224, 224)
    logits, feats = model(dummy)
    print("logits:", logits.shape, "features:", feats.shape)
