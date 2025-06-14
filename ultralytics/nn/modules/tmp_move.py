
import math
import re
from typing import final
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.torch_utils import model_info

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


class Gate(nn.Module):
    def __init__(
        self,
        num_experts: int = 8,
        channels: int = 512,
    ):
        super().__init__()

        self.root = int(math.isqrt(num_experts))
        self.avg_pool = nn.AdaptiveAvgPool2d((self.root, self.root))

        # 使用更大的隐藏层增强表达能力
        hidden_dim = int(num_experts * 2.0)
        self.spatial_mixer = nn.Sequential(
            nn.Linear(num_experts, hidden_dim, bias=True),
            nn.SiLU(),
            # 设置bias增加自由度，不使用bias的话经过sigmoid激活函数后，所有专家的初始权重会在0.5附近
            nn.Linear(hidden_dim, num_experts, bias=True),
            # 绝对不能用 nn.Softmax(dim=-1), 否则性能严重下降，初始权重会被约束在1/num_experts附近，太小了 
            nn.Sigmoid(),  
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        pooled = self.avg_pool(x)  # (B, C, root, root)
        # print(pooled.shape)
        weights = self.spatial_mixer(pooled.view(B, C, -1))  # (B, C, num_experts)
        return weights


class MoVE(nn.Module):
    """
    MoVE: Multi-experts Convolutional Neural Network with Gate mechanism.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 9,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()

        self.num_experts = num_experts

        # 并行化专家计算
        self.expert_conv = nn.Conv2d(
            in_channels,
            in_channels * num_experts,
            kernel_size,
            padding=dilation,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.expert_norm = nn.InstanceNorm2d(in_channels * num_experts)
        self.expert_act = nn.SiLU()

        self.gate = Gate(num_experts=num_experts, channels=in_channels)

        self.out_project = Conv(c1=in_channels, c2=out_channels, k=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        A = self.num_experts

        # 获取门控权重和索引
        weights = self.gate(x)  # (B, C, A)

        # 使用分组卷积处理所有通道
        expert_outputs = self.expert_act(
            self.expert_norm(self.expert_conv(x))
        )  # (B, C*A, H, W)
        expert_outputs = expert_outputs.view(B, C, A, H, W)  # (B, C, A, H, W)

        # 权重应用与求和
        weights = weights.view(B, C, A, 1, 1)
        moe_out = (expert_outputs * weights).sum(dim=2)

        moe_out = self.out_project(moe_out)

        return moe_out

def channel_shuffle(x, groups):
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.size()
    assert num_channels % groups == 0, "num_channels should be " "divisible by groups"
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class MoVE_GhostModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        main_kernel_size: int = 3,  # 主分支的卷积核大小
        num_experts: int = 16,  # 轻量分支专家数量
        cheap_kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()

        self.middle_channels = int(in_channels // 2)
        self.primary_conv = Conv(
            in_channels, self.middle_channels, k=main_kernel_size, d=dilation, act=True
        )

        self.cheap_operation = MoVE(
            self.middle_channels,
            self.middle_channels,
            num_experts,
            cheap_kernel_size,
            dilation,
        )

        self.out_project = Conv(self.middle_channels * 2, out_channels, k=1, act=True)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = channel_shuffle(out, 2)
        out = self.out_project(out)
        return out

class MG_ELAN(nn.Module):
    def  __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_experts: int = 16,
        middle_ratio: float = 0.5,
        num_blocks: int = 2
    ):
        super().__init__()

        middle_channels = int(in_channels * middle_ratio)
        block_channels = int(in_channels * middle_ratio)
        final_channels = int(2 * middle_channels) + int(num_blocks  * block_channels)
        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            internal_block = MoVE_GhostModule(
                in_channels=middle_channels,
                out_channels=block_channels,
                main_kernel_size=kernel_size,
                num_experts=num_experts,
                cheap_kernel_size=3,
                dilation = 1,
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
        middle_ratio: float = 0.5,
    ):
        super().__init__()
        self.channels = channels
        middle_channels = int(channels * middle_ratio)
        self.middle_channels = middle_channels
        self.inv_sqrt_mid = 1 / math.sqrt(channels)

        self.main_conv = Conv(c1=channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=channels, c2=middle_channels, k=1, act=True)

        self.qkv = nn.ModuleDict({
            'W': nn.Conv2d(in_channels=middle_channels, out_channels=1 + 2 * middle_channels, kernel_size=1, bias=True),
            'H': nn.Conv2d(in_channels=middle_channels, out_channels=1 + 2 * middle_channels, kernel_size=1, bias=True)
        })

        self.conv_fusion = nn.ModuleDict({
            'W': light_ConvBlock(in_channels=middle_channels, out_channels=middle_channels),
            'H': light_ConvBlock(in_channels=middle_channels, out_channels=middle_channels)
        })

        final_channels = int(2 * middle_channels)
        self.out_project = Conv(c1=final_channels, c2=channels, k=1, act=True)

    def _apply_axis_attention(self, x, axis):
        """通用轴注意力计算"""
        qkv = self.qkv[axis](x)
        query, key, value = torch.split(qkv, [1, self.middle_channels, self.middle_channels], dim=1)
        
        # 明确指定softmax维度
        dim = -1 if axis == 'W' else -2
        context_scores = F.softmax(query, dim=dim)
        context_vector = (F.silu(key) * context_scores).sum(dim=dim, keepdim=True)
        gate = F.sigmoid(value)
        out = x + gate * context_vector.expand_as(value) # 将全局上下文向量乘以权重，并广播注入到特征图中
        return out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C, H, W]
        
        Returns:
            torch.Tensor: 输出张量，形状为 [B, C, H, W]
        """
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        # 宽轴注意力
        x_W = self._apply_axis_attention(x_main, 'W') 
        x_W_fused = self.conv_fusion['W'](x_W) + x_W
        # 高轴注意力
        x_H = self._apply_axis_attention(x_W_fused, 'H') 
        x_H_fused = self.conv_fusion['H'](x_H) +  x_H

        x_out = torch.cat([x_H_fused, x_short], dim=1)

        x_out = self.out_project(x_out)

        return x_out

# class DualAxisAggAttn(nn.Module):
#     def __init__(
#         self,
#         channels: int,
#         middle_ratio: float = 0.5,
#         alpha_init_value: float = 0.5,
#     ):
#         super().__init__()
#         self.channels = channels
#         middle_channels = int(channels * middle_ratio)
#         self.middle_channels = middle_channels
#         self.inv_sqrt_mid = 1 / math.sqrt(channels)

#         self.main_conv = Conv(c1=channels, c2=middle_channels, k=1, act=True)
#         self.short_conv = Conv(c1=channels, c2=middle_channels, k=1, act=True)

#         self.qkv = nn.ModuleDict({
#             'W': nn.Conv2d(in_channels=middle_channels, out_channels=1 + 2 * middle_channels, kernel_size=1, bias=True),
#             'H': nn.Conv2d(in_channels=middle_channels, out_channels=1 + 2 * middle_channels, kernel_size=1, bias=True)
#         })

#         self.alpha = nn.ParameterDict({
#             'W': nn.Parameter(torch.ones(1) * alpha_init_value),
#             'H': nn.Parameter(torch.ones(1) * alpha_init_value)
#         })

#         self.conv_fusion = nn.ModuleDict({
#             'W': light_ConvBlock(in_channels=middle_channels, out_channels=middle_channels),
#             'H': light_ConvBlock(in_channels=middle_channels, out_channels=middle_channels)
#         })

#         final_channels = int(2 * middle_channels)
#         self.out_project = Conv(c1=final_channels, c2=channels, k=1, act=True)

#     def _apply_axis_attention(self, x, axis):
#         """通用轴注意力计算"""
#         qkv = self.qkv[axis](x)
#         query, key, value = torch.split(qkv, [1, self.middle_channels, self.middle_channels], dim=1)
        
#         # 明确指定softmax维度
#         dim = -1 if axis == 'W' else -2
#         context_scores = F.softmax(query, dim=dim)
#         context_vector = (F.silu(key) * context_scores).sum(dim=dim, keepdim=True)
#         # 将全局上下文向量乘以权重，并广播注入到特征图中
#         gate = F.tanh(self.alpha[axis] * value) # 使用tanh函数的原因是因为这里是逐元素激活，保留元素的负值可以保留模型的表达能力
#         out = x + gate * context_vector.expand_as(value) 
#         return out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C, H, W]
        
        Returns:
            torch.Tensor: 输出张量，形状为 [B, C, H, W]
        """
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        # 宽轴注意力
        x_W = self._apply_axis_attention(x_main, 'W') 
        x_W_fused = self.conv_fusion['W'](x_W) + x_W
        # 高轴注意力
        x_H = self._apply_axis_attention(x_W_fused, 'H') 
        x_H_fused = self.conv_fusion['H'](x_H) +  x_H

        x_out = torch.cat([x_H_fused, x_short], dim=1)

        x_out = self.out_project(x_out)

        return x_out

class TransMoVE(nn.Module):
    """采用ELAN结构"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 9,
        kernel_size: int = 3,
    ):
        super().__init__()
        # 注意力子层
        # --------------------------------------------------------------
        self.attn = DualAxisAggAttn(channels=in_channels)

        # 局部特征提取模块
        #  --------------------------------------------------------------
        self.local_extractor = MG_ELAN(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            num_experts=num_experts,
            middle_ratio=0.5,
            num_blocks=2
        )


        # 输出映射层
        # --------------------------------------------------------------
        if out_channels != in_channels:
            self.out_project = Conv(c1=in_channels, c2=out_channels, k=1, act=True)
        else:
            self.out_project = nn.Identity()
     
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 注意力子层
        residual = x
        x = self.attn(x) +  residual

        # 局部特征提取子层
        residual = x
        x = self.local_extractor(x) + residual

        x = self.out_project(x)
        
        return  x

if __name__ == "__main__":
    model = TransMoVE(in_channels=64, out_channels=64)
    model_info(model,detailed=False)
    x = torch.randn(1, 64, 224, 224)
    # 将模型设置为评估模式
    model.eval()
    y = model(x)
    print(y.shape)


