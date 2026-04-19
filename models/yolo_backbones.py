import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class MobileNetV3Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetV3Backbone, self).__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        mobilenet = mobilenet_v3_large(weights=weights).features

        self.stage1 = nn.Sequential(*list(mobilenet.children())[:7])
        self.stage2 = nn.Sequential(*list(mobilenet.children())[7:13])
        self.stage3 = nn.Sequential(*list(mobilenet.children())[13:])
        self.out_channels = [40, 112, 960]

    def forward(self, x):
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        return [c3, c4, c5]


class ConvBNSiLU(nn.Module):
    def __init__(self, in_c, out_c, k=1, s=1, p=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class CSPBlock(nn.Module):
    def __init__(self, in_c, out_c, hidden_c):
        super().__init__()
        self.conv1 = ConvBNSiLU(in_c, hidden_c)
        self.conv2 = ConvBNSiLU(in_c, hidden_c)
        self.res_block = nn.Sequential(
            ConvBNSiLU(hidden_c, hidden_c, 3, 1, 1),
            ConvBNSiLU(hidden_c, hidden_c, 3, 1, 1)
        )
        self.conv3 = ConvBNSiLU(hidden_c * 2, out_c)

    def forward(self, x):
        y1 = self.res_block(self.conv1(x))
        y2 = self.conv2(x)
        return self.conv3(torch.cat([y1, y2], dim=1))


class CSPDarknetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBNSiLU(3, 32, 3, 2, 1)
        self.stage1 = nn.Sequential(ConvBNSiLU(32, 64, 3, 2, 1), CSPBlock(64, 64, 32))
        self.stage2 = nn.Sequential(ConvBNSiLU(64, 128, 3, 2, 1), CSPBlock(128, 128, 64))
        self.stage3 = nn.Sequential(ConvBNSiLU(128, 256, 3, 2, 1), CSPBlock(256, 256, 128))
        self.stage4 = nn.Sequential(ConvBNSiLU(256, 512, 3, 2, 1), CSPBlock(512, 512, 256))
        self.out_channels = [128, 256, 512]

    def forward(self, x):
        x = self.stage1(self.stem(x))
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return [c3, c4, c5]
