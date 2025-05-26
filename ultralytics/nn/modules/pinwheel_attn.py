import math
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
import torch.nn.functional as F

class PinwheelAttnBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        self.qkv_W = Conv(c1=channels, c2=1+(2*channels), k=1, act=True)
        self.qkv_H = Conv(c1=channels, c2=1+(2*channels), k=1, act=True)

        self.filter = nn.Sequential(
            Conv(c1=channels, c2=channels, k=1, act=True),
            Conv(
                c1=channels, c2=channels, k=3, g=channels, act=True
            ),
            Conv(c1=channels, c2=channels, k=1, act=True),
        )

    def forward(self, x):
        # 横向选择性聚合全局上下文信息
        qkv_W = self.qkv_W(x)
        query, key, value = torch.split(qkv_W, [1, self.channels, self.channels], dim=1)
        context_scores = F.softmax(query, dim=-1)
        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)
        x_W = x + value * context_vector.expand_as(x) / math.sqrt(self.channels)

        # 纵向选择性聚合全局上下文信息
        qkv_H = self.qkv_H(x)
        query, key, value = torch.split(qkv_H, [1, self.channels, self.channels], dim=1)
        context_scores = F.softmax(query, dim=-2)
        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-2, keepdim=True)
        x_H = x + value * context_vector.expand_as(x) / math.sqrt(self.channels)

        att_out = x_W + x_H

        x_out = self.filter(att_out)

        return x_out


class PinwheelAttn(nn.Module):
    """风车注意力，哈哈哈哈哈"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
    ):
        super().__init__()

        middle_channels = int(in_channels * 0.5)
        self.middle_channels = middle_channels

        
        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.blocks = nn.ModuleList([
            PinwheelAttnBlock(channels=middle_channels)
            for _ in range(num_blocks)
        ])

        final_conv_in_channels = (2 + num_blocks) * middle_channels

        self.final_conv = Conv(
            c1=final_conv_in_channels, c2=out_channels, k=1, act=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        residual_final = x

        x_short = self.short_conv(x)
        x_main = self.main_conv(x) # 

        y = [x_short, x_main]
        y.extend(block(y[-1]) for block in self.blocks)
    
        # 多路径信息融合
        x_final = torch.cat(y, dim=1)

        return self.final_conv(x_final) + residual_final


if __name__ == "__main__":

    x = torch.randn(4, 64, 80, 80)
    model = PinwheelAttn(64, 64)
    y = model(x)
    print(y.shape)