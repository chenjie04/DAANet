import math
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
import torch.nn.functional as F

class PinwheelAttn(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        self.qk = nn.ModuleDict({
            'W': nn.Conv2d(in_channels=channels, out_channels=1+(1*channels), kernel_size=1, bias=True),
            'H': nn.Conv2d(in_channels=channels, out_channels=1+(1*channels), kernel_size=1, bias=True)
        })
        self.v = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=True)

        self.out_proj = Conv(c1=channels, c2=channels, k=1, act=True)

    def _apply_axis_aggragation(self, x, axis):
        """通用轴特征聚合计算"""
        qk = self.qk[axis](x)
        query, key = torch.split(
            qk, [1, self.channels], dim=1
        )

        # 明确指定softmax维度
        dim = -1 if axis == "W" else -2
        context_scores = F.softmax(query, dim=dim)
        context_vector = (key * context_scores).sum(dim=dim, keepdim=True)

        return context_vector

    def forward(self, x):
        # 横向选择性聚合全局上下文信息
        context_vector_W = self._apply_axis_aggragation(x, axis="W") # (B, C, H, 1)
        
        # 纵向选择性聚合全局上下文信息
        context_vector_H = self._apply_axis_aggragation(x, axis="H") # (B, C, 1, W)

        # 整体上下文信息
        context = context_vector_W * context_vector_H
       
        # 通过门控机制调节上下文信息，并注入到原始特征中 
        out = x + F.sigmoid(self.v(x)) * context

        out = self.out_proj(out)

        return out

class PinwheelAttnBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.attn = PinwheelAttn(channels)
        self.norm1 = nn.BatchNorm2d(channels)

        self.local_extractor = Conv(c1=channels, c2=channels, k=3, g=1, act=True)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        x = self.norm1(x)
        x = self.attn(x)  + residual

        residual = x
        x = self.norm2(x)
        x = self.local_extractor(x)  + residual

        return x

class PinwheelAttnLayer(nn.Module):
    """风车注意力，哈哈哈哈哈"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
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