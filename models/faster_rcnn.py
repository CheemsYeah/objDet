# models/faster_rcnn.py
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN as TV_FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from models.backbone import ResNetBackbone


class FasterRCNN_ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained_backbone=False, backbone_name="resnet50"):
        super(FasterRCNN_ResNet50, self).__init__()

        # 1. 实例化我们统一的 ResNet-50 Backbone
        # 注意：torchvision 的 FasterRCNN 要求 backbone 包含 out_channels 属性
        self.backbone = ResNetBackbone(
            backbone_name=backbone_name,
            output_type='feature_map',
            pretrained=pretrained_backbone,
        )
        # 在特征图上再加一层 1x1 卷积，平滑特征（可选，但推荐）
        self.backbone_with_conv = nn.Sequential(
            self.backbone,
            nn.Conv2d(self.backbone.out_channels, 256, kernel_size=1)
        )
        self.backbone_with_conv.out_channels = 256  # 重置输出通道以匹配后续网络

        # 2. 定义 RPN 的锚框生成器 (Anchor Generator)
        # 在特征图的每个像素点生成不同面积和比例的锚框 (Anchors)
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        # 3. 定义 RoI Pooling 层 (这里官方采用了更先进的 RoI Align)
        # 从特征图上裁剪 RPN 选出的框，统一大小为 7x7
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],  # 因为 backbone 输出没有字典结构，默认名称为 '0'
            output_size=7,
            sampling_ratio=2
        )

        # 4. 组装最终的 Faster R-CNN 端到端模型
        # 它内部集成了 RPN 网络、NMS 过滤、以及类似 Fast R-CNN 的分类回归头
        self.model = TV_FasterRCNN(
            backbone=self.backbone_with_conv,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

    def forward(self, images, targets=None):
        """
        images: List[Tensor] 或 [B, 3, H, W]
        targets: 包含真实框 bbox 和 labels 的字典 (训练时需要)
        """
        # 如果传入 targets，返回的是 RPN Loss 和 Fast R-CNN Loss 的总和
        # 如果不传入 targets (推理阶段)，返回的是预测的 boxes, labels, scores
        return self.model(images, targets)
