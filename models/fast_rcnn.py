# models/fast_rcnn.py
import torch
import torch.nn as nn
from torchvision.ops import roi_pool
from models.backbone import ResNetBackbone


class FastRCNN(nn.Module):
    def __init__(self, num_classes, spatial_scale=1.0 / 32, pretrained_backbone=False, backbone_name="resnet50"):
        super(FastRCNN, self).__init__()
        self.num_classes = num_classes
        # ResNet50 layer4 将原图缩小了 32 倍
        self.spatial_scale = spatial_scale
        self.roi_size = (7, 7)  # RoI Pooling 输出尺寸

        # 1. 统一的骨干网络 (设置输出为 feature_map，处理整张图)
        self.backbone = ResNetBackbone(
            backbone_name=backbone_name,
            output_type='feature_map',
            pretrained=pretrained_backbone,
        )

        # 2. 全连接层特征展开
        self.fc = nn.Sequential(
            nn.Linear(self.backbone.out_channels * self.roi_size[0] * self.roi_size[1], 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 3. 分类与回归头
        self.classifier = nn.Linear(1024, self.num_classes)
        self.bbox_regressor = nn.Linear(1024, self.num_classes * 4)

    def forward(self, images, rois):
        """
        images: [B, 3, H, W] - 完整的图像批次
        rois: List[Tensor], 每个 Tensor 形状为 [N_i, 4]
              是 Selective Search 给出的候选框坐标 (x1, y1, x2, y2)，对应原图尺度
        """
        # 1. 一次性提取全图特征图，大大节省计算量！
        feature_maps = self.backbone(images)  # [B, 2048, H/32, W/32]

        # 2. 转换 RoIs 格式以适配 torchvision.ops.roi_pool
        # roi_pool 需要的格式是 [K, 5]，第一列是 batch 的 index，后四列是坐标
        rois_with_indices = []
        for batch_idx, roi in enumerate(rois):
            batch_indices = torch.full((roi.size(0), 1), batch_idx, dtype=roi.dtype, device=roi.device)
            rois_with_indices.append(torch.cat((batch_indices, roi), dim=1))
        rois_tensor = torch.cat(rois_with_indices, dim=0)  # [总框数, 5]

        # 3. RoI Pooling (将不同尺寸的候选框映射到特征图上，并池化为 7x7)
        pooled_features = roi_pool(feature_maps, rois_tensor, self.roi_size, self.spatial_scale)
        # pooled_features 形状: [总框数, 2048, 7, 7]

        # 4. 展平并经过 FC 层
        flattened = pooled_features.view(pooled_features.size(0), -1)
        fc_features = self.fc(flattened)

        # 5. 输出分类与回归结果
        cls_scores = self.classifier(fc_features)
        bbox_deltas = self.bbox_regressor(fc_features)

        return cls_scores, bbox_deltas
