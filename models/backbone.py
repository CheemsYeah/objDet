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
