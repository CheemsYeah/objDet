import torch
import torch.nn as nn

from models.backbone import ResNetBackbone


class DETRDetector(nn.Module):
    def __init__(
        self,
        num_classes,
        num_queries=100,
        hidden_dim=256,
        pretrained_backbone=False,
        backbone_name="resnet50",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.backbone = ResNetBackbone(
            backbone_name=backbone_name,
            output_type="feature_map",
            pretrained=pretrained_backbone,
        )
        self.conv = nn.Conv2d(self.backbone.out_channels, hidden_dim, kernel_size=1)

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
        )

        self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.backbone(x)
        h = self.conv(features)
        batch_size, _, height, width = h.shape

        pos = torch.cat(
            [
                self.col_embed[:width].unsqueeze(0).repeat(height, 1, 1),
                self.row_embed[:height].unsqueeze(1).repeat(1, width, 1),
            ],
            dim=-1,
        ).flatten(0, 1).unsqueeze(1)

        src = h.flatten(2).permute(2, 0, 1)
        target = self.query_pos.unsqueeze(1).repeat(1, batch_size, 1)
        hs = self.transformer(src + pos, target)

        out_class = self.linear_class(hs)
        out_bbox = self.linear_bbox(hs)
        return out_class.transpose(0, 1), out_bbox.transpose(0, 1)


class DETR_ResNet50(DETRDetector):
    pass
