import importlib
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

DEFAULT_PRETRAINED_DIR = Path(
    os.environ.get(
        "BODYMAP_IDENTITY_PRETRAINED_DIR",
        "/home/shnh/DATA/zjy/BodyMAP_identity_pretrained",
    )
)
CONVNEXT_V2_BASE_IN1K = DEFAULT_PRETRAINED_DIR / "convnextv2_base.fcmae_ft_in22k_in1k.safetensors"
CONVNEXT_V2_BASE_22K = DEFAULT_PRETRAINED_DIR / "convnextv2_base_22k_224_ema.pt"


def resize_with_padding(x, size=224):
    """Resize BCHW input without distorting the pressure-map aspect ratio."""
    height, width = x.shape[-2:]
    scale = size / max(height, width)
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    x = F.interpolate(
        x,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    pad_height = size - resized_height
    pad_width = size - resized_width
    return F.pad(
        x,
        (
            pad_width // 2,
            pad_width - pad_width // 2,
            pad_height // 2,
            pad_height - pad_height // 2,
        ),
    )


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class PressureCNN(nn.Module):
    """Pressure-native CNN that retains coarse body-contact geometry."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 8 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class PressureEmbeddingNet(nn.Module):
    """Pressure encoder for metric learning and open-set matching."""

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 4)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 4, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        embedding = self.projection(self.features(x))
        return F.normalize(embedding, p=2, dim=1)


class ArcMarginProduct(nn.Module):
    """ArcFace angular-margin classifier; labels are only used during training."""

    def __init__(self, embedding_dim: int, num_classes: int, scale=30.0, margin=0.3):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.margin_correction = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels=None):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale
        sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=0.0))
        target_cosine = cosine * self.cos_margin - sine * self.sin_margin
        target_cosine = torch.where(
            cosine > self.threshold,
            target_cosine,
            cosine - self.margin_correction,
        )
        one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).to(cosine.dtype)
        logits = one_hot * target_cosine + (1.0 - one_hot) * cosine
        return logits * self.scale


class ResNet18Identity(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        if x.shape[-2:] != (224, 224):
            x = resize_with_padding(x)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


class TimmIdentity(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained_dir: str | Path | None = None,
        pretrained: bool = True,
    ):
        super().__init__()
        timm = importlib.import_module("timm")

        pretrained_root = Path(pretrained_dir) if pretrained_dir is not None else DEFAULT_PRETRAINED_DIR
        local_state, local_source = self._load_local_state(model_name, pretrained_root)

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained and local_state is None,
            num_classes=num_classes,
        )
        # The checkpoint was trained with ImageNet-normalized RGB input.
        # Repeating a pressure map to three channels without normalization
        # shifts the pretrained feature distribution and can prevent learning.
        pretrained_cfg = self.backbone.pretrained_cfg
        mean = pretrained_cfg.get("mean", (0.485, 0.456, 0.406))
        std = pretrained_cfg.get("std", (0.229, 0.224, 0.225))
        self.register_buffer("input_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        if local_state is not None:
            missing, unexpected = self.backbone.load_state_dict(local_state, strict=False)
            print(
                f"Loaded local {model_name} checkpoint from {local_source}. "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

    @staticmethod
    def _load_local_state(model_name: str, pretrained_root: Path):
        if model_name != "convnextv2_base":
            return None, None

        in1k_path = pretrained_root / CONVNEXT_V2_BASE_IN1K.name
        in22k_path = pretrained_root / CONVNEXT_V2_BASE_22K.name
        if in1k_path.exists():
            safetensors_torch = importlib.import_module("safetensors.torch")
            state = safetensors_torch.load_file(in1k_path)
            return strip_classifier_head(state), in1k_path
        if in22k_path.exists():
            checkpoint = torch.load(in22k_path, map_location="cpu")
            state = checkpoint.get("model", checkpoint)
            return strip_classifier_head(state), in22k_path
        return None, None

    def forward(self, x):
        if x.shape[-2:] != (224, 224):
            x = resize_with_padding(x)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.input_mean) / self.input_std
        return self.backbone(x)


def strip_classifier_head(state):
    classifier_prefixes = ("head.fc.", "head.")
    classifier_keys = {"fc.weight", "fc.bias", "classifier.weight", "classifier.bias"}
    return {
        key: value
        for key, value in state.items()
        if not key.startswith(classifier_prefixes) and key not in classifier_keys
    }


def build_model(name: str, num_classes: int, embedding_dim: int = 256):
    if name == "small_cnn":
        return SmallCNN(num_classes)
    if name == "pressure_cnn":
        return PressureCNN(num_classes)
    if name == "pressure_arcface":
        return PressureEmbeddingNet(embedding_dim)
    if name == "resnet18":
        return ResNet18Identity(num_classes)
    if name == "convnextv2_base":
        return TimmIdentity("convnextv2_base", num_classes)
    if name.startswith("timm:"):
        return TimmIdentity(name.split(":", 1)[1], num_classes)
    raise ValueError(f"Unknown model {name}")
