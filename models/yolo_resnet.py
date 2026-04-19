# models/yolo_resnet.py
import torch
import torch.nn as nn

from models.backbone import ResNetBackbone
from models.yolo_backbones import CSPDarknetBackbone, MobileNetV3Backbone


def build_yolo_backbone(backbone_name, pretrained_backbone):
    if backbone_name in {"resnet18", "resnet34", "resnet50"}:
        backbone = ResNetBackbone(
            backbone_name=backbone_name,
            output_type="feature_map",
            pretrained=pretrained_backbone,
        )
        return backbone, backbone.out_channels
    if backbone_name == "mobilenetv3":
        backbone = MobileNetV3Backbone(pretrained=pretrained_backbone)
        return backbone, backbone.out_channels[-1]
    if backbone_name == "cspdarknet":
        backbone = CSPDarknetBackbone()
        return backbone, backbone.out_channels[-1]
    raise ValueError(f"Unsupported YOLO backbone: {backbone_name}")


class YOLOBaseline(nn.Module):
    def __init__(self, num_classes, num_bboxes=2, pretrained_backbone=False, backbone_name="resnet50"):
        super(YOLOBaseline, self).__init__()
        self.num_classes = num_classes
        self.num_bboxes = num_bboxes
        self.S = 7

        self.backbone, backbone_out_channels = build_yolo_backbone(backbone_name, pretrained_backbone)

        self.spatial_reducer = nn.Sequential(
            nn.Conv2d(backbone_out_channels, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )
        self.grid_pool = nn.AdaptiveAvgPool2d((self.S, self.S))

        self.out_channels = self.num_bboxes * 5 + self.num_classes
        self.yolo_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, self.out_channels, kernel_size=1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]

        feat = self.spatial_reducer(feat)
        feat = self.grid_pool(feat)

        out = self.yolo_head(feat)
        out = out.permute(0, 2, 3, 1)
        return out
