import math
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
import torch.nn.functional as F


class PinwheelAttn(nn.Module):
    """风车注意力，哈哈哈哈哈"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        middle_channels = int(in_channels * 0.5)
        self.middle_channels = middle_channels

        final_conv_in_channels = 3 * middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.qkv_W = Conv(c1=middle_channels, c2=1 + middle_channels, k=1, act=True)
        self.qkv_H = Conv(c1=middle_channels, c2=1 + middle_channels, k=1, act=True)

        self.filter = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
            Conv(
                c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True
            ),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )


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

        # 横向选择性聚合全局上下文信息
        qkv_W = self.qkv_W(x_main)
        query, key = torch.split(qkv_W, [1, self.middle_channels], dim=1)
        context_scores = F.softmax(query, dim=-1)
        context_vector = key * context_scores
        context_vector_W = torch.sum(context_vector, dim=-1, keepdim=True)

        # 纵向选择性聚合全局上下文信息
        qkv_H = self.qkv_H(x_main)
        query, key = torch.split(qkv_H, [1, self.middle_channels], dim=1)
        context_scores = F.softmax(query, dim=-2)
        context_vector = key * context_scores
        context_vector_H = torch.sum(context_vector, dim=-2, keepdim=True)

        # [B, C, H, 1] @ [B, C, 1, W] -> [B, C, H, W]
        context = torch.matmul(context_vector_W, context_vector_H) / math.sqrt(self.middle_channels)
        context = self.filter(context)

        # 多路径信息融合
        x_final = torch.cat((context, x_main, x_short), dim=1)

        return self.final_conv(x_final) + residual_final


if __name__ == "__main__":

    x = torch.randn(4, 64, 80, 80)
    model = PinwheelAttn(64, 64)
    y = model(x)
    print(y.shape)