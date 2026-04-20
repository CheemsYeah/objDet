# models/backbone.py
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)


SUPPORTED_RESNET_BACKBONES = ("resnet18", "resnet34", "resnet50")
SUPPORTED_YOLO_BACKBONES = ("resnet18", "resnet34", "resnet50", "mobilenetv3", "cspdarknet")

# Keep backbone recommendations close to model definitions so train/inference
# can share one source of truth. These defaults are speed-first.
MODEL_RECOMMENDED_BACKBONES = {
    "rcnn": "resnet18",
    "fast_rcnn": "resnet18",
    "faster_rcnn": "resnet18",
    "yolo": "mobilenetv3",
    "ssd": "resnet18",
    "detr": "resnet18",
}

MODEL_BACKBONE_CHOICES = {
    "rcnn": SUPPORTED_RESNET_BACKBONES,
    "fast_rcnn": SUPPORTED_RESNET_BACKBONES,
    "faster_rcnn": SUPPORTED_RESNET_BACKBONES,
    "yolo": SUPPORTED_YOLO_BACKBONES,
    "ssd": SUPPORTED_RESNET_BACKBONES,
    "detr": SUPPORTED_RESNET_BACKBONES,
}


BACKBONE_CONFIGS = {
    "resnet18": {
        "builder": resnet18,
        "weights": ResNet18_Weights.DEFAULT,
        "out_channels": 512,
    },
    "resnet34": {
        "builder": resnet34,
        "weights": ResNet34_Weights.DEFAULT,
        "out_channels": 512,
    },
    "resnet50": {
        "builder": resnet50,
        "weights": ResNet50_Weights.DEFAULT,
        "out_channels": 2048,
    },
}


class ResNetBackbone(nn.Module):
    def __init__(self, backbone_name="resnet50", output_type="feature_map", pretrained=False):
        super().__init__()
        if backbone_name not in BACKBONE_CONFIGS:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        config = BACKBONE_CONFIGS[backbone_name]
        weights = config["weights"] if pretrained else None
        resnet = config["builder"](weights=weights)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.output_type = output_type
        self.backbone_name = backbone_name
        self.out_channels = config["out_channels"]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feat = self.features(x)
        if self.output_type == "vector":
            feat = self.pool(feat)
            feat = feat.view(feat.size(0), -1)
        return feat


class ResNet50Backbone(ResNetBackbone):
    def __init__(self, output_type="feature_map", pretrained=False):
        super().__init__(backbone_name="resnet50", output_type=output_type, pretrained=pretrained)
