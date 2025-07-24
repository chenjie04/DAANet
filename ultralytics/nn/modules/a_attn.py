import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k2



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
                "W": nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=True),
                "H": nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=True),
            }
        )


    def _apply_axis_attention(self, x, axis):
        """通用轴注意力计算"""
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
        out = x + gate * context_vector.expand_as(value)

        out = self.linear_fusion[axis](out)

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

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

class AttnModule(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio=1, drop_path=0.
    ):
        super().__init__()
        self.channels = channels
        self.attn = DualAxisAggAttn(channels)
        self.f1 = ConvBN(channels, mlp_ratio * channels, 1, with_bn=False)
        self.f2 = ConvBN(channels, mlp_ratio * channels, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * channels, channels, 1, with_bn=True)
        self.dwconv2 = ConvBN(channels, channels, 7, 1, (7 - 1) // 2, groups=channels, with_bn=False)
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.attn(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


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
            internal_block = AttnModule(channels=middle_channels,)

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