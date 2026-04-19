# models/ssd_resnet.py
import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)


class SSD_ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained_backbone=False, backbone_name="resnet50"):
        super(SSD_ResNet50, self).__init__()
        self.num_classes = num_classes

        if backbone_name == "resnet18":
            builder = resnet18
            weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
            layer3_channels = 256
            layer4_channels = 512
        elif backbone_name == "resnet34":
            builder = resnet34
            weights = ResNet34_Weights.DEFAULT if pretrained_backbone else None
            layer3_channels = 256
            layer4_channels = 512
        elif backbone_name == "resnet50":
            builder = resnet50
            weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
            layer3_channels = 1024
            layer4_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        resnet = builder(weights=weights)
        self.backbone_layer3 = nn.Sequential(*list(resnet.children())[:7])
        self.backbone_layer4 = resnet.layer4

        self.extras = nn.ModuleList([
            nn.Sequential(nn.Conv2d(layer4_channels, 512, 1), nn.ReLU(), nn.Conv2d(512, 1024, 3, stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(1024, 256, 1), nn.ReLU(), nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(512, 128, 1), nn.ReLU(), nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(256, 128, 1), nn.ReLU(), nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU())
        ])

        mbox_cfg = [4, 6, 6, 6, 4, 4]
        in_channels = [layer3_channels, layer4_channels, 1024, 512, 256, 256]

        self.loc_heads = nn.ModuleList()
        self.conf_heads = nn.ModuleList()
        for in_c, num_boxes in zip(in_channels, mbox_cfg):
            self.loc_heads.append(nn.Conv2d(in_c, num_boxes * 4, kernel_size=3, padding=1))
            self.conf_heads.append(nn.Conv2d(in_c, num_boxes * self.num_classes, kernel_size=3, padding=1))

    def forward(self, x):
        features = []
        x = self.backbone_layer3(x)
        features.append(x)

        x = self.backbone_layer4(x)
        features.append(x)

        for layer in self.extras:
            x = layer(x)
            features.append(x)

        locs, confs = [], []
        for feat, loc_head, conf_head in zip(features, self.loc_heads, self.conf_heads):
            locs.append(loc_head(feat).permute(0, 2, 3, 1).contiguous().view(feat.size(0), -1, 4))
            confs.append(conf_head(feat).permute(0, 2, 3, 1).contiguous().view(feat.size(0), -1, self.num_classes))

        return torch.cat(locs, dim=1), torch.cat(confs, dim=1)
