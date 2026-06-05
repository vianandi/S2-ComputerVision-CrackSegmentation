import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import DualAttention, AttentionGate
from models.aspp import ASPP


class ResidualBlock(nn.Module):
    """Residual Block with skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class EncoderBlock(nn.Module):
    """Encoder block with Residual + Dual Attention"""
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels)
        self.use_attention = use_attention
        if use_attention:
            self.attention = DualAttention(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.res_block(x)
        if self.use_attention:
            x = self.attention(x)
        skip = x
        x = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    """Decoder block with Residual + Dual Attention"""
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True, skip_attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.res_block = ResidualBlock(in_channels + skip_channels, out_channels)
        self.use_attention = use_attention
        self.skip_attention = skip_attention
        if use_attention:
            self.attention = DualAttention(out_channels)
        if skip_attention:
            self.att_gate = AttentionGate(in_channels, skip_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        if self.skip_attention:
            skip = self.att_gate(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        if self.use_attention:
            x = self.attention(x)
        return x


class DualAttentionResidualUNet(nn.Module):
    """Dual-Attention Residual U-Net with ASPP for Fine-Grained Crack Segmentation"""
    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        base_channels=64,
        deep_supervision=False,
        edge_supervision=False,
        skip_attention=False,
        aspp_dilations=None,
        aspp_use_global_pool=True,
        aspp_use_strip_pool=False,
        aspp_dropout=0.5,
        use_attention=True,
        use_aspp=True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.edge_supervision = edge_supervision
        self.use_attention = use_attention
        self.use_aspp = use_aspp
        
        # Initial convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.enc1 = EncoderBlock(base_channels, base_channels, use_attention=use_attention)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2, use_attention=use_attention)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4, use_attention=use_attention)
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8, use_attention=use_attention)
        
        # Bottleneck with ASPP
        bottleneck_layers = [ResidualBlock(base_channels * 8, base_channels * 8)]
        if use_aspp:
            bottleneck_layers.append(
                ASPP(
                    base_channels * 8,
                    base_channels * 8,
                    dilations=aspp_dilations,
                    use_global_pool=aspp_use_global_pool,
                    use_strip_pool=aspp_use_strip_pool,
                    drop_rate=aspp_dropout,
                )
            )
        if use_attention:
            bottleneck_layers.append(DualAttention(base_channels * 8))
        self.bottleneck = nn.Sequential(*bottleneck_layers)
        
        # Decoder
        self.dec4 = DecoderBlock(
            base_channels * 8,
            base_channels * 8,
            base_channels * 4,
            use_attention=use_attention,
            skip_attention=skip_attention,
        )
        self.dec3 = DecoderBlock(
            base_channels * 4,
            base_channels * 4,
            base_channels * 2,
            use_attention=use_attention,
            skip_attention=skip_attention,
        )
        self.dec2 = DecoderBlock(
            base_channels * 2,
            base_channels * 2,
            base_channels,
            use_attention=use_attention,
            skip_attention=skip_attention,
        )
        self.dec1 = DecoderBlock(
            base_channels,
            base_channels,
            base_channels,
            use_attention=use_attention,
            skip_attention=skip_attention,
        )
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, num_classes, 1)
        )

        if self.deep_supervision:
            self.side4 = nn.Conv2d(base_channels * 4, num_classes, 1)
            self.side3 = nn.Conv2d(base_channels * 2, num_classes, 1)
            self.side2 = nn.Conv2d(base_channels, num_classes, 1)

        if self.edge_supervision:
            self.edge_head = nn.Sequential(
                nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, num_classes, 1)
            )
    
    def forward(self, x):
        input_size = x.shape[2:]
        # Input
        x = self.input_conv(x)
        
        # Encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        d4 = self.dec4(x, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)
        
        # Output (logits)
        logits = self.output(d1)

        if self.deep_supervision or self.edge_supervision:
            outputs = {
                "logits": logits,
                "deep": [],
                "edge": None,
            }

            if self.deep_supervision:
                side4 = F.interpolate(self.side4(d4), size=input_size, mode='bilinear', align_corners=False)
                side3 = F.interpolate(self.side3(d3), size=input_size, mode='bilinear', align_corners=False)
                side2 = F.interpolate(self.side2(d2), size=input_size, mode='bilinear', align_corners=False)
                outputs["deep"] = [side4, side3, side2]

            if self.edge_supervision:
                outputs["edge"] = self.edge_head(skip1)

            return outputs

        return logits
