import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

import logging

logger = logging.getLogger(__name__)

USE_FLASH_ATTN = False
try:
    import torch

    if (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):  # Ampere or newer
        from flash_attn.flash_attn_interface import flash_attn_func

        USE_FLASH_ATTN = True
    else:
        from torch.nn.functional import scaled_dot_product_attention as sdpa

        logger.warning(
            "FlashAttention is not available on this device. Using scaled_dot_product_attention instead."
        )
except Exception:
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    logger.warning(
        "FlashAttention is not available on this device. Using scaled_dot_product_attention instead."
    )


class SelfAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(  # type: ignore
                q.contiguous().half(), k.contiguous().half(), v.contiguous().half()
            ).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = v @ attn.transpose(-2, -1)

            x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x)


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
            nn.Linear(num_experts, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts, bias=True),
            nn.Sigmoid(),  # 绝对不能用 nn.Softmax(dim=-1), 否则性能严重下降
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
    ):
        super().__init__()

        self.num_experts = num_experts
        padding = kernel_size // 2

        # 并行化专家计算
        self.expert_conv = nn.Conv2d(
            in_channels,
            in_channels * num_experts,
            kernel_size,
            padding=padding,
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


    
class MoVE_no_gate(nn.Module):
    """
    MoVE: Multi-experts Convolutional Neural Network with Gate mechanism.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 9,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.num_experts = num_experts
        padding = kernel_size // 2

        # 并行化专家计算
        self.expert_conv = nn.Conv2d(
            in_channels,
            in_channels * num_experts,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.expert_norm = nn.InstanceNorm2d(in_channels * num_experts)
        self.expert_act = nn.SiLU()


        self.out_project = Conv(c1=in_channels, c2=out_channels, k=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        A = self.num_experts

        # 使用分组卷积处理所有通道
        expert_outputs = self.expert_act(
            self.expert_norm(self.expert_conv(x))
        )  # (B, C*A, H, W)

        expert_outputs = expert_outputs.view(B, C, A, H, W)  # (B, C, A, H, W)

        moe_out = expert_outputs.sum(dim=2)

        moe_out = self.out_project(moe_out)

        return moe_out
    
class MoVE_scale(nn.Module):
    """
    MoVE: Multi-experts Convolutional Neural Network with Gate mechanism.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 9,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.num_experts = num_experts
        padding = kernel_size // 2

        # 并行化专家计算
        self.expert_conv = nn.Conv2d(
            in_channels,
            in_channels * num_experts,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.expert_norm = nn.InstanceNorm2d(in_channels * num_experts)
        self.expert_act = nn.SiLU()

        self.scales = nn.Parameter(torch.ones(in_channels * num_experts), requires_grad=True)


        self.out_project = Conv(c1=in_channels, c2=out_channels, k=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        A = self.num_experts

        # 使用分组卷积处理所有通道
        expert_outputs = self.expert_act(
            self.expert_norm(self.expert_conv(x))
        )  # (B, C*A, H, W)
        expert_outputs = expert_outputs * self.scales.view(1, -1, 1, 1)
        expert_outputs = expert_outputs.view(B, C, A, H, W)  # (B, C, A, H, W)

        moe_out = expert_outputs.sum(dim=2)

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


class GhostModule(nn.Module):
    """
        主分支kernel size为1：
        YOLO113n summary (fused): 130 layers, 2,061,428 parameters, 0 gradients, 5.2 GFLOPs
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:08<00:00,  4.48it/s]
                       all       4952      12032      0.801      0.708       0.79      0.591
                 aeroplane        204        285      0.876      0.767      0.873      0.648
                   bicycle        239        337      0.874      0.813      0.886      0.677
                      bird        282        459      0.807      0.654      0.762      0.531
                      boat        172        263      0.731      0.624      0.707      0.474
                    bottle        212        469      0.846      0.527      0.659       0.43
                       bus        174        213      0.847      0.765      0.841      0.725
                       car        721       1201      0.887      0.828      0.904      0.724
                       cat        322        358      0.828      0.788      0.846      0.667
                     chair        417        756       0.75      0.458      0.595      0.394
                       cow        127        244      0.693      0.803      0.819      0.619
               diningtable        190        206      0.728      0.704       0.74      0.576
                       dog        418        489      0.767      0.728      0.811      0.619
                     horse        274        348      0.856      0.819      0.895      0.705
                 motorbike        222        325      0.852      0.762      0.877      0.642
                    person       2007       4528      0.893      0.736      0.859      0.594
               pottedplant        224        480      0.681      0.405      0.519      0.283
                     sheep         97        242      0.772      0.713      0.771      0.586
                      sofa        223        239      0.631      0.724      0.757      0.627
                     train        259        282      0.867      0.858        0.9      0.701
                 tvmonitor        229        308      0.836      0.678      0.783      0.601
    Speed: 0.1ms preprocess, 0.7ms inference, 0.0ms loss, 0.3ms postprocess per image
    Results saved to runs/yolo11_VOC/113n139
        主分支kernel size为3：
        YOLO113n summary (fused): 130 layers, 2,409,588 parameters, 0 gradients, 6.1 GFLOPs
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:08<00:00,  4.43it/s]
                       all       4952      12032      0.811      0.714      0.801      0.601
                 aeroplane        204        285       0.87      0.777      0.876      0.672
                   bicycle        239        337      0.867      0.813      0.898      0.689
                      bird        282        459      0.829       0.63      0.754      0.531
                      boat        172        263      0.734      0.639      0.707       0.47
                    bottle        212        469      0.864      0.515      0.671      0.442
                       bus        174        213      0.835      0.781      0.858      0.745
                       car        721       1201      0.894      0.813       0.91       0.73
                       cat        322        358      0.856       0.81      0.874      0.689
                     chair        417        756      0.774      0.475      0.623      0.421
                       cow        127        244      0.737      0.766      0.833      0.632
               diningtable        190        206      0.752      0.684      0.757      0.598
                       dog        418        489      0.817      0.749      0.829      0.634
                     horse        274        348      0.815      0.868      0.903      0.719
                 motorbike        222        325       0.85      0.783      0.878      0.649
                    person       2007       4528      0.906      0.738      0.869      0.603
               pottedplant        224        480      0.673       0.41      0.506      0.275
                     sheep         97        242      0.752      0.781      0.816      0.616
                      sofa        223        239      0.682      0.757      0.787      0.636
                     train        259        282      0.881      0.838      0.887      0.688
                 tvmonitor        229        308      0.841      0.659      0.776      0.582
    Speed: 0.1ms preprocess, 0.7ms inference, 0.0ms loss, 0.2ms postprocess per image
    Results saved to runs/yolo11_VOC/113n140

        采用channel shuffle：
    YOLO113n summary (fused): 130 layers, 2,409,588 parameters, 0 gradients, 6.1 GFLOPs
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:08<00:00,  4.39it/s]
                       all       4952      12032      0.806      0.715      0.801      0.602
                 aeroplane        204        285      0.868      0.784      0.867      0.659
                   bicycle        239        337      0.885      0.843        0.9      0.687
                      bird        282        459       0.78       0.66      0.759      0.522
                      boat        172        263      0.697      0.655      0.722      0.461
                    bottle        212        469      0.865      0.494      0.671      0.443
                       bus        174        213      0.838      0.737      0.855      0.752
                       car        721       1201      0.899      0.829      0.912      0.738
                       cat        322        358      0.849      0.799      0.869      0.687
                     chair        417        756      0.756      0.479      0.616      0.421
                       cow        127        244      0.705      0.811      0.833      0.631
               diningtable        190        206      0.747      0.665      0.731      0.584
                       dog        418        489      0.794      0.736      0.836       0.64
                     horse        274        348      0.857      0.845      0.912      0.721
                 motorbike        222        325      0.862      0.818      0.905      0.664
                    person       2007       4528       0.89      0.756      0.868      0.596
               pottedplant        224        480      0.697      0.403      0.517      0.277
                     sheep         97        242      0.772      0.769      0.809      0.617
                      sofa        223        239       0.64      0.757      0.761      0.639
                     train        259        282      0.874      0.833       0.89      0.687
                 tvmonitor        229        308      0.847      0.633      0.785      0.602
    Speed: 0.1ms preprocess, 0.7ms inference, 0.0ms loss, 0.2ms postprocess per image
    Results saved to runs/yolo11_VOC/113n141

        增加输出映射层：
    YOLO113n summary (fused): 134 layers, 2,497,108 parameters, 0 gradients, 6.3 GFLOPs
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:09<00:00,  4.33it/s]
                       all       4952      12032      0.801      0.734      0.808      0.605
                 aeroplane        204        285      0.895      0.782      0.883      0.659
                   bicycle        239        337      0.907      0.816      0.898      0.685
                      bird        282        459      0.773      0.683      0.769      0.541
                      boat        172        263      0.732      0.654      0.715      0.461
                    bottle        212        469      0.855      0.577      0.706       0.47
                       bus        174        213      0.833      0.784      0.862      0.755
                       car        721       1201      0.881      0.833      0.913       0.73
                       cat        322        358      0.858      0.813      0.878      0.689
                     chair        417        756      0.751      0.528      0.634      0.424
                       cow        127        244      0.714      0.807      0.829      0.627
               diningtable        190        206      0.757      0.709      0.769      0.614
                       dog        418        489      0.802      0.744      0.842       0.66
                     horse        274        348      0.843      0.853      0.892      0.699
                 motorbike        222        325      0.838        0.8      0.881      0.646
                    person       2007       4528      0.896       0.75      0.867        0.6
               pottedplant        224        480       0.69      0.438      0.526      0.292
                     sheep         97        242      0.719       0.81      0.833      0.625
                      sofa        223        239      0.621      0.753      0.772      0.645
                     train        259        282      0.843      0.858      0.893      0.678
                 tvmonitor        229        308      0.811      0.683      0.802      0.609
    Speed: 0.1ms preprocess, 0.7ms inference, 0.0ms loss, 0.2ms postprocess per image
    Results saved to runs/yolo11_VOC/113n142

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,  # 主分支的卷积核大小
        num_experts: int = 1,  # 轻量分支专家数量
    ):
        super().__init__()

        self.middle_channels = int(in_channels // 2)
        self.primary_conv = Conv(in_channels, self.middle_channels, k=kernel_size, act=True)

        self.cheap_operation = Conv(
            self.middle_channels,
            self.middle_channels,
            k=3,
            g=self.middle_channels,
            act=True,
        )  # 3x3深度分离卷积, 即num_experts=1

        self.out_project = Conv(self.middle_channels * 2, out_channels, k=1, act=True)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = channel_shuffle(out, 2)
        out = self.out_project(out)
        return out


class MoVE_GhostModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,  # 主分支的卷积核大小
        num_experts: int = 16,  # 轻量分支专家数量
    ):
        super().__init__()

        self.middle_channels = int(in_channels // 2)
        self.primary_conv = Conv(
            in_channels, self.middle_channels, k=kernel_size, act=True
        )

        self.cheap_operation = MoVE(
            self.middle_channels,
            self.middle_channels,
            num_experts,
            3,  # 轻量分支卷积核大小一般都设为3，用于替代3x3深度分离卷积
        )

        self.out_project = Conv(self.middle_channels * 2, out_channels, k=1, act=True)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = channel_shuffle(out, 2)
        out = self.out_project(out)
        return out

class DualAxisAggAttn(nn.Module):
    """Efficient dual-axis aggregation attention module.DAANet的基本模块"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        middle_channels = int(in_channels * 0.5)
        self.middle_channels = middle_channels

        final_conv_in_channels = 2 * middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)


        self.qkv_W = Conv(c1=middle_channels, c2=1+(2 * middle_channels), k=1, act=True)
        self.conv_W = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
            Conv(
                c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True
            ),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )


        self.qkv_H = Conv(c1=middle_channels, c2=1+(2 * middle_channels), k=1, act=True)
        self.conv_H = nn.Sequential(
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
        qkv_W = self.qkv_W(x_main) # [B, C, H, W] -> [B, 1+2C, H, W]
        # query: [B, 1, H, W]
        # key, value: [B, C, H, W]
        query, key, value = torch.split(qkv_W, [1, self.middle_channels, self.middle_channels], dim=1)
        # 沿着维度W应用softmax
        context_scores = F.softmax(query, dim=-1)
        # 计算上下文向量
        # [B, C, H, W] x [B, 1, H, W] -> [B, C, H, W]
        context_vector = key * context_scores
        # [B, C, H, W] -> [B, C, H, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)
        # [B, C, H, W] x [B, C, H, 1] -> [B, C, H, W]
        x_W = value * context_vector.expand_as(value) / math.sqrt(self.middle_channels) + x_main

        # 信息过滤
        x_W = self.conv_W(x_W) +  x_W

        # 纵向选择性聚合全局上下文信息
        qkv_H = self.qkv_H(x_W)
        query, key, value = torch.split(qkv_H, [1, self.middle_channels, self.middle_channels], dim=1)
        # 沿着维度H应用softmax
        context_scores = F.softmax(query, dim=-2)
        context_vector = key * context_scores
        # [B, C, H, W] -> [B, C, 1, W]
        context_vector = torch.sum(context_vector, dim=-2, keepdim=True)
        x_H = value * context_vector.expand_as(value) / math.sqrt(self.middle_channels) + x_W

        # 信息过滤
        x_H = self.conv_H(x_H) + x_H

        # 多路径信息融合
        x_final = torch.cat((x_H, x_short), dim=1)

        return self.final_conv(x_final) + residual_final


class FakeDualAxisAggAttn(nn.Module):
    """ """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        middle_channels = int(in_channels * 0.5)

        final_conv_in_channels = 2 * middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.conv_W = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
            Conv(
                c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True
            ),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )

        self.conv_H = nn.Sequential(
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
        x_main = self.main_conv(x)

        x_plus_W = self.conv_W(x_main) +  x_main

        x_plus_WH = self.conv_H(x_plus_W) + x_plus_W

        # 多路径信息融合
        x_final = torch.cat((x_plus_WH, x_short), dim=1)

        return self.final_conv(x_final) + residual_final


class TransMoVE(nn.Module):
    """采用ELAN结构
        COCO
     YOLO113n summary: 343 layers, 2,609,872 parameters, 2,609,856 gradients, 7.0 GFLOPs
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.406
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.568
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.436
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.209
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.446
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.589
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.329
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.546
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.602
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.380
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.666
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.789
    Results saved to runs/yolo11_coco/113n2

    """

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
        self.spatial_attn = DualAxisAggAttn(out_channels, out_channels)

        # self attention
        # ---------------------------------------------------------------
        # YOLO113n summary: 307 layers, 2,648,576 parameters, 2,648,560 gradients, 7.0 GFLOPs
        # print(f"TransMoVE: {in_channels} -> {out_channels}")
        # num_heads = max(1, out_channels // 64)
        # self.spatial_attn = SelfAttn(dim=out_channels,num_heads=num_heads, area=1)

        # Area attention
        # ---------------------------------------------------------------
        # YOLO113n summary: 307 layers, 2,648,576 parameters, 2,648,560 gradients, 7.0 GFLOPs
        # num_heads = max(1, out_channels // 64)
        # area = int(512 /  out_channels)
        # if area < 4:
        #     area = 1
        # if out_channels == 256:
        #     area = 1
        # else:
        #     area = 4
        # # 主干网络的最后一个阶段不能划分，不然推理阶段由于图像分辨率不能整除而报错
        # self.spatial_attn = SelfAttn(dim=out_channels,num_heads=num_heads, area=area)

        # Criss Cross Attention
        # YOLO113n summary: 287 layers, 2,415,808 parameters, 2,415,792 gradients, 6.6 GFLOPs
        # ---------------------------------------------------
        # from ultralytics.nn.modules.criss_cross_attn import CrissCrossAttention
        # self.spatial_attn = CrissCrossAttention(in_channels)

        # ELAN结构
        # ---------------------------------------------------------------
        num_blocks = 2
        num_in_block = 1
        middle_ratio = 0.5

        self.num_blocks = num_blocks

        middle_channels = int(out_channels * middle_ratio)
        block_channels = int(out_channels * middle_ratio)
        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if num_in_block == 1:
                internal_block = MoVE_GhostModule(
                    in_channels=middle_channels,
                    out_channels=block_channels,
                    num_experts=num_experts,
                    kernel_size=kernel_size,
                )
                # internal_block = GhostModule(
                #     in_channels=middle_channels,
                #     out_channels=block_channels,
                #     num_experts=num_experts,
                #     kernel_size=kernel_size,
                # )
            else:
                internal_block = []
                for _ in range(num_in_block):
                    internal_block.append(
                        MoVE_GhostModule(
                            in_channels=middle_channels,
                            out_channels=block_channels,
                            num_experts=num_experts,
                            kernel_size=kernel_size,
                        )
                    )
                internal_block = nn.Sequential(*internal_block)

            self.blocks.append(internal_block)

        final_conv_in_channels = (
            num_blocks * block_channels + int(out_channels * middle_ratio) * 2
        )
        self.final_conv = Conv(
            c1=final_conv_in_channels, c2=out_channels, k=1, act=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 注意力子层
        residual = x
        x = residual + self.spatial_attn(x)

        # ELAN子层
        residual = x
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        block_outs = []
        x_block = x_main
        for block in self.blocks:
            x_block = block(x_block)
            block_outs.append(x_block)
        x_final = torch.cat((*block_outs[::-1], x_main, x_short), dim=1)
        return self.final_conv(x_final) + residual
