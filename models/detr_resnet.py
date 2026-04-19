# models/detr_resnet.py
import torch
import torch.nn as nn
from models.backbone import ResNetBackbone


class DETR_ResNet50(nn.Module):
    def __init__(self, num_classes, num_queries=100, hidden_dim=256, pretrained_backbone=False, backbone_name="resnet50"):
        super(DETR_ResNet50, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        # 1. 统一的 ResNet-50 骨干网络
        self.backbone = ResNetBackbone(
            backbone_name=backbone_name,
            output_type='feature_map',
            pretrained=pretrained_backbone,
        )

        # 2. 通道降维：将 ResNet50 的 2048 维降到 Transformer 的 256 维
        self.conv = nn.Conv2d(self.backbone.out_channels, hidden_dim, kernel_size=1)

        # 3. 核心：标准 Transformer
        # (实际官方源码中包含了更复杂的位置编码和注意力层，这里采用 PyTorch 自带接口做概念实现)
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048
        )

        # 4. 可学习的 Object Queries (替代了传统检测算法中的 Anchors)
        # 大小为 [查询数量, 隐藏层维度]
        self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))

        # 5. 可学习的空间位置编码 (针对图像的 2D 位置编码)
        # 假设最大输入产生的特征图不超过 50x50
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

        # 6. 分类与回归前馈层 (FFN)
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)  # +1 代表背景类
        self.linear_bbox = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 预测归一化的 cx, cy, w, h
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. CNN 提取特征
        features = self.backbone(x)  # [B, 2048, H, W]
        h = self.conv(features)  # [B, 256, H, W]
        B, C, H, W = h.shape

        # 2. 生成 2D 绝对位置编码并与特征图大小对齐
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)  # [H*W, 1, 256]

        # 3. 展平图像特征，并送入 Transformer
        # Transformer 要求输入形状为 [Sequence_length, Batch, Dim]
        src = h.flatten(2).permute(2, 0, 1)  # [H*W, B, 256]

        # Object Queries 扩展到当前 Batch
        target = self.query_pos.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, 256]

        # 注意：Transformer 内部是将 src + pos 作为 Encoder 的输入
        # 这里为了简化，直接叠加送入 (官方源码在每一层都有叠加机制)
        hs = self.transformer(src + pos, target)  # 输出 [num_queries, B, 256]

        # 4. 预测输出
        out_class = self.linear_class(hs)  # [100, B, num_classes+1]
        out_bbox = self.linear_bbox(hs)  # [100, B, 4]

        return out_class.transpose(0, 1), out_bbox.transpose(0, 1)  # 返回 [B, 100, ...]
