import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # Use GroupNorm instead of BatchNorm (more stable for small batches)
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        pool = self.gap(x)
        pool = F.interpolate(pool, size=size, mode='bilinear', align_corners=False)
        return pool


class ASPPStripPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPStripPooling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        # Pool along height and width to capture long cracks
        pool_h = x.mean(dim=2, keepdim=True)
        pool_w = x.mean(dim=3, keepdim=True)

        pool_h = self.conv(pool_h)
        pool_w = self.conv(pool_w)

        pool_h = F.interpolate(pool_h, size=size, mode='bilinear', align_corners=False)
        pool_w = F.interpolate(pool_w, size=size, mode='bilinear', align_corners=False)
        return pool_h + pool_w


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            # Use GroupNorm instead of BatchNorm
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=256,
        dilations=None,
        use_global_pool=True,
        use_strip_pool=False,
        drop_rate=0.5,
    ):
        super(ASPP, self).__init__()

        if dilations is None:
            dilations = [1, 3, 6, 9]

        modules = []
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        ))

        # Dilated convolutions
        for dilation in dilations[1:]:
            modules.append(ASPPConv(in_channels, out_channels, dilation))

        # Optional global average pooling
        if use_global_pool:
            modules.append(ASPPPooling(in_channels, out_channels))

        # Optional strip pooling for elongated structures
        if use_strip_pool:
            modules.append(ASPPStripPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)