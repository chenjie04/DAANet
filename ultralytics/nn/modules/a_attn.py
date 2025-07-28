import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k2

class light_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            Conv(c1=in_channels, c2=in_channels, k=1, act=True),
            Conv(c1=in_channels, c2=in_channels, k=3, g=in_channels, act=True),
            Conv(c1=in_channels, c2=out_channels, k=1, act=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)
    
class DualAxisAggAttn(nn.Module):
    def __init__(
        self,
        channels: int,
    ):
        super().__init__()
        self.channels = channels
        
        self.qkv = nn.ModuleDict(
            {
                "W": nn.Conv2d(
                    in_channels=channels,
                    out_channels=1 + 2 * channels,
                    kernel_size=1,
                    bias=True,
                ),
                "H": nn.Conv2d(
                    in_channels=channels,
                    out_channels=1 + 2 * channels,
                    kernel_size=1,
                    bias=True,
                ),
            }
        )

        self.linear_fusion = nn.ModuleDict(
            {
                "W": Conv(c1=channels, c2=channels, k=3, g=channels, act=True),
                "H": Conv(c1=channels, c2=channels, k=3, g=channels, act=True),
            }
        )


    def _apply_axis_attention(self, x, axis):
        """通用轴注意力计算"""
        residual = x
        qkv = self.qkv[axis](x)
        query, key, value = torch.split(
            qkv, [1, self.channels, self.channels], dim=1
        )

        # 明确指定softmax维度
        dim = -1 if axis == "W" else -2
        context_scores = F.softmax(query, dim=dim)
        context_vector = (key * context_scores).sum(dim=dim, keepdim=True)
        # gate = F.tanh(self.alpha[axis] * value) # 效果不及sigmoid
        # gate = F.silu(value) # 效果最差
        gate = F.sigmoid(value)
        # 将全局上下文向量乘以权重，并广播注入到特征图中
        out = residual + gate * context_vector.expand_as(value)

        residual = out
        out = residual + self.linear_fusion[axis](out)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C, H, W]

        Returns:
            torch.Tensor: 输出张量，形状为 [B, C, H, W]
        """
        
        # 宽轴注意力
        x_W = self._apply_axis_attention(x, "W")
        
        # 高轴注意力
        x_H = self._apply_axis_attention(x_W, "H")
   

        return x_H

def spatial_channels_attention(x: torch.Tensor) -> torch.Tensor:
    # 聚合全局特征
    context = F.adaptive_avg_pool2d(x, 1)
    # 计算每个位置与全局特征之间的相似度
    logits = x * context
    # 归一化
    weights = F.sigmoid(logits)
    # 乘以权重
    out = x * weights
    return  out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 通过卷积生成空间注意力
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 先进行空间卷积
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化

        # 将池化结果拼接在一起
        x_out = torch.cat([avg_pool, max_pool], dim=1)
        # 通过卷积产生空间注意力图
        attention = self.conv(x_out)
        attention = self.sigmoid(attention)

        # 将注意力图应用于输入
        return x * attention


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

        self.spatial_attn = SpatialAttention(kernel_size=7)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        x = spatial_channels_attention(x) # 60.4
        # x = self.spatial_attn(x) # 60.2
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class AELAN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        middle_ratio: float = 0.5,
        num_blocks: int = 2,
    ):
        super().__init__()

        middle_channels = int(in_channels * middle_ratio)
        block_channels = int(in_channels * middle_ratio)
        final_channels = int(2 * middle_channels) + int(num_blocks * block_channels)
        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            internal_block = Bottleneck(
                c1=middle_channels,
                c2=block_channels,
                shortcut=True,
                g=1,
                k=(3, 1),
                e=0.5,
            )

            middle_channels = block_channels
            self.blocks.append(internal_block)

        self.out_project = Conv(c1=final_channels, c2=out_channels, k=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        block_outs = []
        x_block = x_main
        for block in self.blocks:
            x_block = block(x_block)
            block_outs.append(x_block)
        x_final = torch.cat((*block_outs[::-1], x_main, x_short), dim=1)
        return self.out_project(x_final)   