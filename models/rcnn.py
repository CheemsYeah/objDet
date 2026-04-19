# models/rcnn.py
import torch.nn as nn
from models.backbone import ResNetBackbone


class RCNN(nn.Module):
    def __init__(self, num_classes, pretrained_backbone=False, backbone_name="resnet50"):
        super(RCNN, self).__init__()
        # num_classes 包含背景类 (如 VOC: 20+1=21)
        self.num_classes = num_classes

        # 1. 统一的骨干网络 (设置输出为 vector，因为输入已经是裁剪好的候选框)
        self.backbone = ResNetBackbone(
            backbone_name=backbone_name,
            output_type='vector',
            pretrained=pretrained_backbone,
        )

        # 2. 分类头 (Classification Head)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.out_channels, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_classes)
        )

        # 3. 边界框回归头 (Bounding Box Regression Head)
        # 每个类别独立预测 4 个坐标偏移量 (dx, dy, dw, dh)
        self.bbox_regressor = nn.Sequential(
            nn.Linear(self.backbone.out_channels, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_classes * 4)
        )

    def forward(self, cropped_rois):
        """
        cropped_rois: [N, 3, 224, 224]
        N 是从原图中利用 Selective Search 裁剪并缩放后的候选框图像批次
        """
        # 对每一个裁剪出的候选框提取特征 (此处存在巨大的重复计算！)
        features = self.backbone(cropped_rois)  # [N, 2048]

        # 分类与回归
        cls_scores = self.classifier(features)  # [N, num_classes]
        bbox_deltas = self.bbox_regressor(features)  # [N, num_classes * 4]

        return cls_scores, bbox_deltas
