import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ultralytics.nn.modules import Conv

class Gate2(nn.Module):
    """
    采用softmax函数，experts=16
    YOLO113n summary: 154 layers, 2,616,548 parameters, 0 gradients, 6.6 GFLOPs                                                                                                                                       
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:11<00:00,  3.26it/s]                                                                         
                   all       4952      12032      0.811      0.724      0.809      0.609                                                                                                                          
             aeroplane        204        285      0.902      0.793      0.878      0.653                                                                                                                          
               bicycle        239        337      0.895      0.819      0.904      0.686                                                                                                                          
                  bird        282        459      0.827      0.699      0.785      0.556                                                                                                                          
                  boat        172        263      0.728      0.658      0.739       0.49                                                                                                                          
                bottle        212        469      0.885      0.541      0.697      0.474                                                                                                                          
                   bus        174        213       0.85      0.784      0.858      0.752                                                                                                                          
                   car        721       1201      0.906      0.823      0.911      0.731                                                                                                                          
                   cat        322        358       0.85      0.825      0.876      0.703                                                                                                                          
                 chair        417        756      0.759      0.511      0.645      0.432                                                                                                                          
                   cow        127        244      0.735      0.803      0.832      0.632
           diningtable        190        206      0.728      0.688      0.757      0.595
                   dog        418        489       0.78      0.754      0.842      0.647
                 horse        274        348      0.842      0.859      0.912      0.728
             motorbike        222        325      0.874      0.797      0.885      0.658                 
                person       2007       4528      0.905      0.741      0.874      0.605                 
           pottedplant        224        480      0.693      0.438      0.531      0.292                 
                 sheep         97        242       0.73      0.736      0.798      0.608                 
                  sofa        223        239      0.613      0.774      0.784       0.65                 
                 train        259        282      0.868      0.837      0.884      0.682                 
             tvmonitor        229        308      0.855      0.601      0.785      0.608                 
Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.2ms postprocess per image                        
Results saved to runs/yolo11_VOC/113n143  

使用sigmoid函数，experts=16， 效果也不理想，证明gate2无用
    YOLO113n summary: 154 layers, 2,616,548 parameters, 0 gradients, 6.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:11<00:00,  3.28it/s]
                   all       4952      12032       0.82      0.718      0.809       0.61
             aeroplane        204        285      0.912      0.779      0.878      0.661
               bicycle        239        337      0.878      0.811      0.901      0.687
                  bird        282        459      0.814      0.638      0.772      0.539
                  boat        172        263      0.746      0.635      0.735       0.48
                bottle        212        469      0.871      0.548      0.698      0.467
                   bus        174        213      0.871       0.77      0.862       0.76
                   car        721       1201      0.895      0.842      0.918      0.738
                   cat        322        358      0.834       0.81      0.875      0.691
                 chair        417        756       0.79      0.484      0.622      0.423
                   cow        127        244      0.702       0.82      0.822      0.631
           diningtable        190        206      0.751      0.732      0.781       0.62
                   dog        418        489      0.807       0.76       0.85      0.659
                 horse        274        348      0.852      0.851      0.907      0.731
             motorbike        222        325      0.869      0.797      0.887      0.657
                person       2007       4528      0.904      0.733      0.868      0.603
           pottedplant        224        480      0.731      0.408       0.55      0.308
                 sheep         97        242      0.781      0.727      0.801      0.608
                  sofa        223        239      0.667      0.762      0.778      0.647
                 train        259        282      0.858      0.837      0.883      0.678
             tvmonitor        229        308      0.868      0.623      0.795      0.617
Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to runs/yolo11_VOC/113n144
    """
    def __init__(
        self,
        channels: int = 512,
        groups: int = 1,
    ):
        super().__init__()

        self.avg_pool =nn.AdaptiveAvgPool2d(1)

        self.channels_mixer = nn.Conv2d(channels, channels, kernel_size=1, groups=groups, bias=True)

        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        pooled = self.avg_pool(x)  
        weights = self.channels_mixer(pooled)  # (B, C*num_experts, 1, 1)
        weights = weights.view(B, self.groups, C//self.groups, 1, 1)
        # weights = F.softmax(weights, dim=2)  # 验证sigmoid，Gate1是sigmoid效果很好，softmax效果很差
        weights = F.sigmoid(weights)
        return weights


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
        weights = self.spatial_mixer(pooled.view(B, C, -1))  # (B, C, num_experts)
        return weights


class MoVE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 8,
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

        self.gate = Gate(num_experts)

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
        kernel_size: int = 3,  # 主分支的卷积核大小
        num_experts: int = 8,  # 轻量分支专家数量
    ):
        super().__init__()

        self.middle_channels = int(in_channels // 2)

        self.primary_conv = Conv(
            in_channels, self.middle_channels, k=kernel_size, act=True
        )

        self.cheap_operation = MoVE(
            self.middle_channels, self.middle_channels, num_experts, 3 # 轻量分支卷积核大小一般都设为3，用于替代3x3深度分离卷积
        )

        self.out_project = Conv(self.middle_channels * 2, out_channels, k=1, act=True)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = channel_shuffle(out, 2)
        out = self.out_project(out)
        return out


class ESAA(nn.Module):
    """Efficient Spatial Aggregated Attention with Value Transform"""

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.attn_W = Conv(c1=channels, c2=1, k=1, act=True)
        self.attn_H = Conv(c1=channels, c2=1, k=1, act=True)

        # Value projection layer
        self.v_conv = MoVE_GhostModule(channels, channels, kernel_size=1, num_experts=9)

        self.out_project = MoVE_GhostModule(channels, channels, kernel_size=1, num_experts=9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_final = x  # Store original input for final residual connection

        # Compute Value representation once
        v = self.v_conv(x)

        # Width Attention Path
        residual_w = v
        logits_W = self.attn_W(v)
        context_scores_W = F.softmax(logits_W, dim=-1)
        context_vector_W = (v * context_scores_W).sum(-1, keepdim=True)
        x_W = residual_w + context_vector_W.expand_as(v)

        # Height Attention Path
        residual_h = x_W
        logits_H = self.attn_H(x_W)
        context_scores_H = F.softmax(logits_H, dim=-2)
        context_vector_H = (x_W * context_scores_H).sum(-2, keepdim=True)
        x_H = residual_h + context_vector_H.expand_as(x_W)

        out = v + x_W + x_H
        out = self.out_project(out) + residual_final

        return out


class TransMoVE(nn.Module):
    def __init__(
        self,
        channels: int,
        num_experts: int = 9,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.spatial_attn = ESAA(channels)

        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

        self.local_extractor = MoVE_GhostModule(
            channels, channels, num_experts, kernel_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意力子层
        residual = x
        norm_x = self.norm1(x)
        x = residual + self.spatial_attn(norm_x)

        # 前馈子层
        residual = x
        out = residual + self.local_extractor(x)

        return out

class ESAAM(nn.Module):
    """Efficient spatial aggregation attention module.

    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super().__init__()
        
        middle_channels = int(in_channels * 0.5)
        
        final_conv_in_channels = 4 * middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.attn_W = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_W = Conv(c1=middle_channels, c2=middle_channels, k=3, act=True)
        self.attn_H = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_H = Conv(c1=middle_channels, c2=middle_channels, k=3, act=True)

        
        self.final_conv = Conv(
            c1=final_conv_in_channels, c2=out_channels, k=1, act=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)

        x_W = self.conv_W(x_main)
        logits_W = self.attn_W(x_W)
        context_scores_W = F.softmax(logits_W, dim=-1)
        context_vector_W = x_W + (x_W * context_scores_W).sum(-1, keepdim=True).expand_as(x_W)

        x_H = self.conv_H(x_main)
        logits_H = self.attn_H(x_H)
        context_scores_H = F.softmax(logits_H, dim=-2)
        context_vector_H = x_H + (x_H * context_scores_H).sum(-2, keepdim=True).expand_as(x_H)
        x_final = torch.cat((context_vector_H, context_vector_W, x_main, x_short), dim=1)
        
        return self.final_conv(x_final)
    

    import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

# map: 60.9
# class Gate(nn.Module):
#     def __init__(
#         self,
#         num_experts: int = 8,
#         channels: int = 512,
#     ):
#         super().__init__()

#         self.root = int(math.isqrt(num_experts))
#         self.avg_pool = nn.AdaptiveAvgPool2d((self.root, self.root))

#         # 使用更大的隐藏层增强表达能力
#         hidden_dim = int(num_experts * 2.0)
#         self.spatial_mixer = nn.Sequential(
#             nn.Linear(num_experts, hidden_dim, bias=False),
#             nn.SiLU(), 
#             nn.Linear(hidden_dim, num_experts, bias=True),
#             nn.Sigmoid(), # 绝对不能用 nn.Softmax(dim=-1), 否则性能严重下降
#         )
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, C, _, _ = x.shape
#         pooled = self.avg_pool(x)  # (B, C, root, root)
#         weights = self.spatial_mixer(pooled.view(B, C, -1))  # (B, C, num_experts)
#         return weights


# def channel_shuffle(x, groups):
#     """Channel Shuffle operation.

#     This function enables cross-group information flow for multiple groups
#     convolution layers.

#     Args:
#         x (Tensor): The input tensor.
#         groups (int): The number of groups to divide the input tensor
#             in the channel dimension.

#     Returns:
#         Tensor: The output tensor after channel shuffle operation.
#     """

#     batch_size, num_channels, height, width = x.size()
#     assert num_channels % groups == 0, "num_channels should be " "divisible by groups"
#     channels_per_group = num_channels // groups

#     x = x.view(batch_size, groups, channels_per_group, height, width)
#     x = torch.transpose(x, 1, 2).contiguous()
#     x = x.view(batch_size, -1, height, width)

#     return x


# class MoVE(nn.Module):
#     def __init__(
#         self,
#         channels: int,
#         num_experts: int = 8,
#         kernel_size: int = 3,
#     ):
#         super().__init__()

#         self.middle_channels = int(channels // 2)

#         self.intrinsic_conv = Conv(channels, c2=self.middle_channels, k=3, act=True)

#         self.num_experts = num_experts
#         padding = kernel_size // 2

#         self.expert_conv = nn.Conv2d(
#             self.middle_channels,
#             self.middle_channels * num_experts,
#             kernel_size,
#             padding=padding,
#             groups=self.middle_channels,
#             bias=False,
#         )
#         self.expert_norm = nn.InstanceNorm2d(self.middle_channels * num_experts)
#         self.expert_act = nn.SiLU()
#         self.gate = Gate(num_experts)

#         # 混合通道信息
#         self.channel_mixer = Conv(self.middle_channels * 2, channels, k=1, act=True)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, _, H, W = x.shape

#         intrinsic_x = self.intrinsic_conv(x)

#         A = self.num_experts
#         # 获取门控权重
#         weights = self.gate(intrinsic_x)  # (B, C, A)

#         # 使用分组卷积处理所有通道
#         expert_outputs = self.expert_conv(intrinsic_x)
#         expert_outputs = self.expert_norm(expert_outputs)
#         expert_outputs = self.expert_act(expert_outputs) # (B, C * A, H, W)
#         expert_outputs = expert_outputs.view(B, self.middle_channels, A, H, W)  # (B, C, A, H, W)

#         # 权重应用与求和
#         weights = weights.view(B, self.middle_channels, A, 1, 1)
#         moe_out = (expert_outputs * weights).sum(dim=2)

#         out = torch.cat([moe_out, intrinsic_x], dim=1)
#         out = channel_shuffle(out, 2)
#         # 特征增强
#         out = self.channel_mixer(out)

#         return out

# class ESAA(nn.Module):
#     """Efficient Spatial Aggregated Attention with Value Transform"""
#     def __init__(self, channels: int) -> None:
#         super().__init__()

#         self.attn_S = Conv(c1=channels, c2=1, k=1, act=True)
#         self.attn_W = Conv(c1=channels, c2=1, k=1, act=True)
#         self.attn_H = Conv(c1=channels, c2=1, k=1, act=True)

#         # Value projection layer
#         self.v_conv =  Conv(c1=channels, c2=channels, k=1, act=True)

#         self.out_project = nn.Sequential(
#             Conv(c1=channels, c2=int(channels * 1.0), k=1, act=True),
#             Conv(c1=int(channels * 1.0), c2=channels, k=1, act=True),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         residual_final = x # Store original input for final residual connection

#         # Compute Value representation once
#         v = self.v_conv(x)

#         # Spatial Attention Path
#         residual_s = v 
#         logits_S = self.attn_S(v) 
#         spatial_scores = torch.sigmoid(logits_S)
#         x_spatial = spatial_scores * v + residual_s 

#         # Width Attention Path
#         residual_w = x_spatial 
#         logits_W = self.attn_W(x_spatial) 
#         context_scores_W = F.softmax(logits_W, dim=-1)
#         context_vector_W = (x_spatial * context_scores_W).sum(-1, keepdim=True) 
#         x_W = residual_w + context_vector_W.expand_as(x_spatial) 

#         # Height Attention Path
#         residual_h = x_W 
#         logits_H = self.attn_H(x_W) 
#         context_scores_H = F.softmax(logits_H, dim=-2)
#         context_vector_H = (x_W * context_scores_H).sum(-2, keepdim=True) 
#         x_H = residual_h + context_vector_H.expand_as(x_W) 


#         out = x_spatial + x_W + x_H 
#         out = self.out_project(out) + residual_final 

#         return out
    

# class ESAA_Simplified(nn.Module):
#     """
#     ESAA 模块改造版，采用串行轴向注意力确保全局信息传播。
#     """
#     def __init__(self, channels: int) -> None:
#         super().__init__()
#         # 重用注意力分数计算层 (1x1 Conv)
#         self.attn_H = Conv(c1=channels, c2=1, k=1, act=True)
#         self.attn_W = Conv(c1=channels, c2=1, k=1, act=True)
#         # 注意：原始的 attn_S 在此方案中未使用，因为我们专注于全局聚合

#         # Value projection layer (保持不变)
#         self.v_conv = Conv(c1=channels, c2=channels, k=1, act=True)

#         # Output projection layer (保持不变)
#         self.out_project = Conv(c1=channels, c2=channels, k=1, act=True)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         residual_final = x # 存储原始输入以供最后添加

#         # 1. 计算 Value 'v'
#         v = self.v_conv(x) # Shape: (B, C, H, W)

#         # --- 串行全局聚合 ---

#         # 2. 高度(Height)注意力与聚合
#         # 基于 'v' 计算高度注意力分数
#         logits_H = self.attn_H(v) # Shape: (B, 1, H, W)
#         # 沿高度维度应用 Softmax
#         scores_H = F.softmax(logits_H, dim=-2) # Shape: (B, 1, H, W)
#         # 使用高度分数对 'v' 进行加权求和，沿高度维度聚合
#         aggregated_H = (v * scores_H).sum(dim=-2, keepdim=True) # Shape: (B, C, 1, W)
#         # 将聚合后的高度上下文加回到 'v' (或其他融合方式)
#         # 这里使用加法，保留原始信息并加入上下文
#         v_plus_H = v + aggregated_H.expand_as(v) # Shape: (B, C, H, W)

#         # 3. 宽度(Width)注意力与聚合 (使用已包含高度信息的特征)
#         # 基于 'v_plus_H' 计算宽度注意力分数
#         logits_W = self.attn_W(v_plus_H) # Shape: (B, 1, H, W)
#         # 沿宽度维度应用 Softmax
#         scores_W = F.softmax(logits_W, dim=-1) # Shape: (B, 1, H, W)
#         # 使用宽度分数对 'v_plus_H' 进行加权求和，沿宽度维度聚合
#         # 此时聚合的特征已经包含了高度上下文信息
#         aggregated_W = (v_plus_H * scores_W).sum(dim=-1, keepdim=True) # Shape: (B, C, H, 1)
#         # 将聚合后的宽度上下文加回到 'v_plus_H'
#         v_plus_HW = v_plus_H + aggregated_W.expand_as(v) # Shape: (B, C, H, W)
#         # 现在 v_plus_HW 中的每个位置都通过串行聚合包含了全局信息

#         # 4. 输出投影与最终残差连接
#         out = self.out_project(v_plus_HW) + residual_final # Shape: (B, C, H, W)

#         return out

# class TransMoVE(nn.Module):
#     def __init__(
#         self,
#         channels: int,
#         num_experts: int = 9,
#         kernel_size: int = 3,
#     ):
#         super().__init__()

#         self.spatial_attn = ESAA(channels)

#         self.norm1 = nn.GroupNorm(1, channels)
#         self.norm2 = nn.GroupNorm(1, channels)

#         self.moe = MoVE(channels, num_experts, kernel_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # 注意力子层
#         residual = x
#         norm_x = self.norm1(x)
#         x = residual + self.spatial_attn(norm_x)

#         # 前馈子层
#         residual = x
#         out = residual + self.moe(x)

#         return out

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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 8,
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

        self.gate = Gate(num_experts)

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
        kernel_size: int = 3,  # 主分支的卷积核大小
        num_experts: int = 16,  # 轻量分支专家数量
    ):
        super().__init__()

        self.middle_channels = int(in_channels // 2)
        self.primary_conv = Conv(
            in_channels, self.middle_channels, k=kernel_size, act=True
        )

        self.cheap_operation = MoVE(
            self.middle_channels, self.middle_channels, num_experts, 3 # 轻量分支卷积核大小一般都设为3，用于替代3x3深度分离卷积
        )

        self.out_project = Conv(self.middle_channels * 2, out_channels, k=1, act=True)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = channel_shuffle(out, 2)
        out = self.out_project(out)
        return out


class ESAA(nn.Module):
    """Efficient Spatial Aggregated Attention with Value Transform"""

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.attn_W = Conv(c1=channels, c2=1, k=1, act=True)
        self.attn_H = Conv(c1=channels, c2=1, k=1, act=True)

        # Value projection layer
        self.v_conv = MoVE_GhostModule(channels, channels, kernel_size=1, num_experts=9)

        self.out_project = Conv(c1=channels, c2=channels, k=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_final = x  # Store original input for final residual connection

        # Compute Value representation once
        v = self.v_conv(x)

        # Width Attention Path
        residual_w = v
        logits_W = self.attn_W(v)
        context_scores_W = F.softmax(logits_W, dim=-1)
        context_vector_W = (v * context_scores_W).sum(-1, keepdim=True)
        x_W = residual_w + context_vector_W.expand_as(v)

        # Height Attention Path
        residual_h = x_W
        logits_H = self.attn_H(x_W)
        context_scores_H = F.softmax(logits_H, dim=-2)
        context_vector_H = (x_W * context_scores_H).sum(-2, keepdim=True)
        x_H = residual_h + context_vector_H.expand_as(x_W)

        out = v + x_W + x_H
        out = self.out_project(out) + residual_final

        return out

class ESAAM(nn.Module):
    """Efficient spatial aggregation attention module.
    YOLO113n summary: 166 layers, 2,555,428 parameters, 0 gradients, 6.6 GFLOPs                                                                                                                                       
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:11<00:00,  3.29it/s]                                                                         
                   all       4952      12032      0.808      0.735      0.816      0.617                                                                                                                          
             aeroplane        204        285      0.901      0.798      0.891       0.67                                                                                                                          
               bicycle        239        337      0.885       0.84      0.912      0.704                                                                                                                          
                  bird        282        459      0.797      0.675       0.77      0.543                                                                                                                          
                  boat        172        263      0.749      0.677      0.752      0.495                                                                                                                          
                bottle        212        469      0.847      0.569      0.706      0.482                                                                                                                          
                   bus        174        213      0.855      0.798      0.873      0.767
                   car        721       1201      0.883      0.826      0.914      0.738
                   cat        322        358      0.834      0.838       0.88      0.702
                 chair        417        756      0.747      0.496      0.638      0.433
                   cow        127        244      0.741      0.824      0.848       0.64
           diningtable        190        206      0.745      0.728      0.764       0.61                 
                   dog        418        489      0.786      0.753      0.849      0.663                 
                 horse        274        348      0.859      0.874      0.918      0.738                 
             motorbike        222        325      0.876      0.801      0.899      0.656                 
                person       2007       4528      0.891      0.764      0.874      0.607                 
           pottedplant        224        480      0.689       0.39       0.52       0.29                 
                 sheep         97        242        0.7       0.76      0.821      0.634                 
                  sofa        223        239      0.635      0.741      0.786      0.653                 
                 train        259        282      0.879      0.852      0.903      0.691                 
             tvmonitor        229        308      0.871      0.699      0.796      0.615                 
Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.2ms postprocess per image                        
Results saved to runs/yolo11_VOC/113n133 

    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super().__init__()
        
        middle_channels = int(in_channels * 0.5)
        
        final_conv_in_channels = 4 * middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.attn_W = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_W = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )
        self.attn_H = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_H = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
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

        logits_W = self.attn_W(x_main)
        context_scores_W = F.softmax(logits_W, dim=-1)
        x_plus_W = x_main + (x_main * context_scores_W).sum(-1, keepdim=True).expand_as(x_main)
        x_plus_W = self.conv_W(x_plus_W)

        
        logits_H = self.attn_H(x_plus_W)
        context_scores_H = F.softmax(logits_H, dim=-2)
        x_plus_WH = x_plus_W + (x_plus_W * context_scores_H).sum(-2, keepdim=True).expand_as(x_plus_W)
        x_plus_WH = self.conv_H(x_plus_WH)

        x_final = torch.cat((x_plus_WH, x_plus_W, x_main, x_short), dim=1)
        
        return self.final_conv(x_final) + residual_final
    
class ESAAM_v2(nn.Module):
    """Efficient spatial aggregation attention module.
YOLO113n summary: 166 layers, 2,555,428 parameters, 0 gradients, 6.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:11<00:00,  3.30it/s]
                   all       4952      12032       0.81       0.72      0.805      0.607
             aeroplane        204        285      0.917      0.804      0.892      0.667
               bicycle        239        337      0.906      0.829      0.907      0.698
                  bird        282        459      0.828      0.639      0.755      0.538
                  boat        172        263      0.722      0.658      0.732      0.482
                bottle        212        469       0.84      0.536      0.691      0.469
                   bus        174        213      0.842      0.812      0.868      0.761
                   car        721       1201      0.877       0.84      0.918      0.735
                   cat        322        358      0.852      0.804      0.864      0.686
                 chair        417        756      0.741      0.493      0.631      0.432
                   cow        127        244      0.719      0.775      0.829      0.622
           diningtable        190        206      0.733      0.689      0.787      0.604
                   dog        418        489      0.771      0.773      0.833      0.646
                 horse        274        348      0.867      0.836      0.901      0.722
             motorbike        222        325      0.879      0.805      0.883      0.648
                person       2007       4528      0.888      0.761      0.871      0.605
           pottedplant        224        480      0.694      0.375      0.501       0.28
                 sheep         97        242      0.758       0.74      0.802      0.606
                  sofa        223        239      0.672      0.745      0.783      0.649
                 train        259        282      0.867       0.84       0.87      0.683
             tvmonitor        229        308      0.825      0.644      0.786      0.602
Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to runs/yolo11_VOC/113n134
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super().__init__()
        
        middle_channels = int(in_channels * 0.5)
        
        final_conv_in_channels = 4 * middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.attn_W = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_W = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )
        self.attn_H = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_H = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
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

        logits_W = self.attn_W(x_main)
        context_scores_W = F.softmax(logits_W, dim=-1)
        x_plus_W = x_main + x_main.sum(-1, keepdim=True).expand_as(x_main) * context_scores_W
        x_plus_W = self.conv_W(x_plus_W)

        
        logits_H = self.attn_H(x_plus_W)
        context_scores_H = F.softmax(logits_H, dim=-2)
        x_plus_WH = x_plus_W + x_plus_W.sum(-2, keepdim=True).expand_as(x_plus_W) * context_scores_H
        x_plus_WH = self.conv_H(x_plus_WH)

        x_final = torch.cat((x_plus_WH, x_plus_W, x_main, x_short), dim=1)
        
        return self.final_conv(x_final) + residual_final
    
class ESAAM_v3(nn.Module):
    """Efficient spatial aggregation attention module.
    简单测试了一下，感觉也不行
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super().__init__()
        
        middle_channels = int(in_channels * 0.5)
        
        final_conv_in_channels = 4 * middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.attn_W = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_W = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )
        self.attn_H = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_H = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
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

        logits_W = self.attn_W(x_main)
        context_scores_W = F.sigmoid(logits_W)
        x_plus_W = x_main + x_main.sum(-1, keepdim=True).expand_as(x_main) * context_scores_W
        x_plus_W = self.conv_W(x_plus_W)

        
        logits_H = self.attn_H(x_plus_W)
        context_scores_H = F.sigmoid(logits_H)
        x_plus_WH = x_plus_W + x_plus_W.sum(-2, keepdim=True).expand_as(x_plus_W) * context_scores_H
        x_plus_WH = self.conv_H(x_plus_WH)

        x_final = torch.cat((x_plus_WH, x_plus_W, x_main, x_short), dim=1)
        
        return self.final_conv(x_final) + residual_final
    
class ESAAM_v4(nn.Module):
    """Efficient spatial aggregation attention module.
YOLO113n summary: 166 layers, 2,555,916 parameters, 0 gradients, 6.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:11<00:00,  3.28it/s]
                   all       4952      12032      0.814      0.726      0.809      0.612
             aeroplane        204        285      0.897      0.796      0.889      0.665
               bicycle        239        337      0.878      0.816      0.896       0.69
                  bird        282        459       0.83       0.68      0.779      0.555
                  boat        172        263      0.726      0.635      0.721      0.463
                bottle        212        469       0.84       0.57      0.707      0.472
                   bus        174        213      0.858      0.766      0.863      0.767
                   car        721       1201      0.892      0.834       0.91       0.73
                   cat        322        358      0.848      0.809       0.86      0.688
                 chair        417        756      0.758      0.484       0.61      0.416
                   cow        127        244      0.728        0.8      0.838      0.642
           diningtable        190        206      0.758      0.757      0.791      0.623
                   dog        418        489      0.845      0.724      0.853       0.66
                 horse        274        348      0.849      0.868      0.914       0.74
             motorbike        222        325      0.869      0.803      0.885      0.664
                person       2007       4528      0.898      0.761      0.877      0.608
           pottedplant        224        480      0.738      0.435      0.554      0.309
                 sheep         97        242      0.725      0.752       0.79      0.608
                  sofa        223        239      0.638      0.751      0.768      0.636
                 train        259        282       0.85       0.84      0.891      0.693
             tvmonitor        229        308       0.86       0.64      0.788      0.608
Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to runs/yolo11_VOC/113n136
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super().__init__()
        
        middle_channels = int(in_channels * 0.5)
        
        final_conv_in_channels = 4 * middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.attn_W = Conv(c1=middle_channels, c2=2, k=1, act=True)
        self.conv_W = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )
        self.attn_H = Conv(c1=middle_channels, c2=2, k=1, act=True)
        self.conv_H = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
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

        logits = self.attn_W(x_main)
        logits_sum, logits_W = torch.split(
            logits, [1, 1], dim=1
        )
        context_scores_sum = F.softmax(logits_sum, dim=-1)
        context_scores_W = F.sigmoid(logits_W)
        x_plus_W = x_main + (x_main * context_scores_sum).sum(-1, keepdim=True).expand_as(x_main) * context_scores_W
        x_plus_W = self.conv_W(x_plus_W)

        
        logits = self.attn_H(x_plus_W)
        logits_sum, logits_H = torch.split(
            logits, [1, 1], dim=1
        )
        context_scores_sum = F.softmax(logits_sum, dim=-2)
        context_scores_H = F.sigmoid(logits_H)
        x_plus_WH = x_plus_W + (x_plus_W * context_scores_sum).sum(-2, keepdim=True).expand_as(x_plus_W) * context_scores_H
        x_plus_WH = self.conv_H(x_plus_WH)

        x_final = torch.cat((x_plus_WH, x_plus_W, x_main, x_short), dim=1)
        
        return self.final_conv(x_final) + residual_final
    
class ESAAM_v5(nn.Module):
    """Efficient spatial aggregation attention module.
YOLO113n summary: 166 layers, 2,555,916 parameters, 0 gradients, 6.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:11<00:00,  3.28it/s]
                   all       4952      12032      0.804      0.731       0.81       0.61
             aeroplane        204        285      0.893      0.804      0.885      0.671
               bicycle        239        337      0.904      0.822      0.898      0.689
                  bird        282        459      0.819      0.692      0.788      0.541
                  boat        172        263      0.698      0.654      0.728       0.49
                bottle        212        469      0.849      0.552      0.705      0.473
                   bus        174        213      0.844      0.775      0.864      0.766
                   car        721       1201      0.892      0.823      0.908      0.729
                   cat        322        358      0.872      0.813      0.866      0.683
                 chair        417        756      0.736      0.483      0.621      0.426
                   cow        127        244      0.711      0.816      0.829      0.621
           diningtable        190        206      0.711      0.723      0.778      0.599
                   dog        418        489      0.784      0.765      0.853      0.664
                 horse        274        348      0.848      0.865      0.916      0.734
             motorbike        222        325      0.842      0.819      0.885      0.654
                person       2007       4528       0.89      0.752      0.874       0.61
           pottedplant        224        480      0.704      0.435      0.532      0.296
                 sheep         97        242      0.734      0.793       0.81       0.62
                  sofa        223        239      0.636      0.732      0.763      0.636
                 train        259        282      0.872      0.837      0.892       0.69
             tvmonitor        229        308      0.845      0.656      0.803      0.612
Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to runs/yolo11_VOC/113n137
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super().__init__()
        
        middle_channels = int(in_channels * 0.5)
        
        final_conv_in_channels = 4 * middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.attn_W = Conv(c1=middle_channels, c2=2, k=1, act=True)
        self.conv_W = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )
        self.attn_H = Conv(c1=middle_channels, c2=2, k=1, act=True)
        self.conv_H = nn.Sequential(
            Conv(c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True),
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

        logits = self.attn_W(x_main)
        logits_sum, logits_W = torch.split(
            logits, [1, 1], dim=1
        )
        context_scores_sum = F.softmax(logits_sum, dim=-1)
        context_scores_W = F.sigmoid(logits_W)
        x_plus_W = x_main * context_scores_W + (x_main * context_scores_sum).sum(-1, keepdim=True).expand_as(x_main)
        x_plus_W = self.conv_W(x_plus_W)

        
        logits = self.attn_H(x_plus_W)
        logits_sum, logits_H = torch.split(
            logits, [1, 1], dim=1
        )
        context_scores_sum = F.softmax(logits_sum, dim=-2)
        context_scores_H = F.sigmoid(logits_H)
        x_plus_WH = x_plus_W * context_scores_H + (x_plus_W * context_scores_sum).sum(-2, keepdim=True).expand_as(x_plus_W)
        x_plus_WH = self.conv_H(x_plus_WH)

        x_final = torch.cat((x_plus_WH, x_plus_W, x_main, x_short), dim=1)
        
        return self.final_conv(x_final) + residual_final

class TransMoVE(nn.Module):
    def __init__(
        self,
        channels: int,
        num_experts: int = 9,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.spatial_attn = ESAAM(channels, channels)
        
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

        self.local_extractor = MoVE_GhostModule(
            channels, channels, kernel_size, num_experts
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意力子层
        residual = x
        norm_x = self.norm1(x)
        x = residual + self.spatial_attn(norm_x)

        # 前馈子层
        residual = x
        out = residual + self.local_extractor(x)

        return out
    

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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 8,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.num_experts = num_experts
        padding = kernel_size // 2

        # 并行化专家计算
        self.experts = nn.ModuleList(
            [
                Conv(
                    c1=in_channels,
                    c2=in_channels,
                    k=kernel_size,
                    s=1,
                    p=padding,
                    g=in_channels,
                    act=True,
                )
                for _ in range(num_experts)
            ]
        )

        self.gate = Gate(num_experts)

        self.out_project = Conv(c1=in_channels, c2=out_channels, k=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        A = self.num_experts

        # 获取门控权重和索引
        weights = self.gate(x)  # (B, C, A)

        # 使用分组卷积处理所有通道
        expert_outputs = torch.stack(
            [
                expert(x)
                for expert in self.experts
            ],
            dim=1,
        )
        expert_outputs = expert_outputs.view(B, C, A, H, W)

        # 权重应用与求和
        weights = weights.view(B, C, A, 1, 1)
        moe_out = (expert_outputs * weights).sum(dim=2)

        moe_out = self.out_project(moe_out)

        return moe_out


class MoVE(nn.Module):
    """
    玄学，调整了一下代码位置，效果变差了
    YOLO113n summary: 166 layers, 2,555,428 parameters, 0 gradients, 6.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:11<00:00,  3.31it/s]
                   all       4952      12032      0.816      0.721      0.809      0.613
             aeroplane        204        285      0.914      0.781      0.886      0.671
               bicycle        239        337      0.883      0.801      0.898      0.688
                  bird        282        459      0.812      0.678      0.774      0.549
                  boat        172        263      0.739      0.666      0.734      0.488
                bottle        212        469      0.862      0.533      0.702      0.471
                   bus        174        213      0.862      0.765      0.862      0.755
                   car        721       1201      0.898       0.83      0.916      0.734
                   cat        322        358      0.795       0.83      0.877      0.708
                 chair        417        756      0.797      0.461      0.622      0.425
                   cow        127        244      0.723      0.802      0.834      0.637
           diningtable        190        206      0.752      0.663      0.753      0.612
                   dog        418        489      0.834      0.742      0.847      0.654
                 horse        274        348      0.861      0.842      0.909      0.726
             motorbike        222        325      0.867      0.825      0.894      0.673
                person       2007       4528      0.906      0.739       0.87      0.606
           pottedplant        224        480       0.72      0.403      0.529      0.299
                 sheep         97        242       0.72       0.76      0.805      0.621                                                                                                                          
                  sofa        223        239      0.664      0.762      0.772      0.636                                                                                                                          
                 train        259        282      0.874      0.848      0.898      0.691                                                                                                                          
             tvmonitor        229        308      0.844      0.687      0.794       0.61                                                                                                                          
Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.2ms postprocess per image                                                                                                                                 
Results saved to runs/yolo11_VOC/113n147 
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 8,
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

        self.gate = Gate(num_experts=num_experts)

        self.out_project = Conv(c1=in_channels, c2=out_channels, k=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        A = self.num_experts

        # 使用分组卷积处理所有通道
        expert_outputs = self.expert_act(
            self.expert_norm(self.expert_conv(x))
        )  # (B, C*A, H, W)
        expert_outputs = expert_outputs.view(B, C, A, H, W)  # (B, C, A, H, W)

        # 调整了一下expert_outputs和weights的计算顺序，验证鲁棒性
        # 可能因为神经网络有一些in-place操作，会导致权重计算存在细微差异
        # 权重应用与求和
        weights = self.gate(x) # (B, C, A) 
        weights = weights.view(B, C, A, 1, 1)
        moe_out = (expert_outputs * weights).sum(dim=2)

        moe_out = self.out_project(moe_out)

        return moe_out


# 2025-05-16
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
        num_experts: int = 16
        YOLO113n summary: 166 layers, 2,555,428 parameters, 0 gradients, 6.6 GFLOPs
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:11<00:00,  3.29it/s]
                       all       4952      12032      0.808      0.735      0.816      0.617
                 aeroplane        204        285      0.901      0.798      0.891       0.67
                   bicycle        239        337      0.885       0.84      0.912      0.704
                      bird        282        459      0.797      0.675       0.77      0.543
                      boat        172        263      0.749      0.677      0.752      0.495
                    bottle        212        469      0.847      0.569      0.706      0.482
                       bus        174        213      0.855      0.798      0.873      0.767
                       car        721       1201      0.883      0.826      0.914      0.738
                       cat        322        358      0.834      0.838       0.88      0.702
                     chair        417        756      0.747      0.496      0.638      0.433
                       cow        127        244      0.741      0.824      0.848       0.64
               diningtable        190        206      0.745      0.728      0.764       0.61
                       dog        418        489      0.786      0.753      0.849      0.663
                     horse        274        348      0.859      0.874      0.918      0.738
                 motorbike        222        325      0.876      0.801      0.899      0.656
                    person       2007       4528      0.891      0.764      0.874      0.607
               pottedplant        224        480      0.689       0.39       0.52       0.29
                     sheep         97        242        0.7       0.76      0.821      0.634
                      sofa        223        239      0.635      0.741      0.786      0.653
                     train        259        282      0.879      0.852      0.903      0.691
                 tvmonitor        229        308      0.871      0.699      0.796      0.615
    Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.2ms postprocess per image
    Results saved to runs/yolo11_VOC/113n133

    num_experts: int = 9
    YOLO113n summary: 166 layers, 2,537,480 parameters, 0 gradients, 6.5 GFLOPs
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:10<00:00,  3.63it/s]
                       all       4952      12032      0.818      0.723      0.809      0.613
                 aeroplane        204        285      0.885      0.811       0.88      0.653
                   bicycle        239        337      0.887      0.815      0.906      0.694
                      bird        282        459       0.82      0.673      0.781      0.556
                      boat        172        263      0.688      0.658      0.733      0.465
                    bottle        212        469      0.863       0.55      0.705      0.478
                       bus        174        213      0.849      0.779      0.852      0.755
                       car        721       1201      0.901      0.817      0.912      0.736
                       cat        322        358      0.835       0.83      0.883      0.711
                     chair        417        756      0.741      0.469      0.613      0.421
                       cow        127        244      0.753      0.774      0.833      0.632
               diningtable        190        206      0.776       0.69       0.77      0.602
                       dog        418        489      0.811      0.769      0.854      0.666
                     horse        274        348      0.855      0.878      0.915      0.738
                 motorbike        222        325      0.882      0.781      0.869      0.651
                    person       2007       4528      0.892      0.761      0.877      0.612
               pottedplant        224        480      0.763      0.379       0.52      0.296
                     sheep         97        242      0.772      0.769      0.813      0.625
                      sofa        223        239      0.645      0.753      0.774      0.646
                     train        259        282      0.861      0.819      0.882      0.687
                 tvmonitor        229        308      0.882      0.682      0.816      0.628
    Speed: 0.1ms preprocess, 1.1ms inference, 0.0ms loss, 0.2ms postprocess per image
    Results saved to runs/yolo11_VOC/113n149

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

        self.gate = Gate(num_experts)

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
        self.primary_conv = Conv(in_channels, self.middle_channels, k=1, act=True)

        self.cheap_operation = Conv(
            self.middle_channels,
            self.middle_channels,
            k=3,
            g=self.middle_channels,
            act=True,
        )  # 3x3深度分离卷积, 即num_experts=1

        # self.out_project = Conv(self.middle_channels * 2, out_channels, k=1, act=True)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        # out = channel_shuffle(out, 2)
        # out = self.out_project(out)
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
    """Efficient dual-axis aggregation attention module.DAANet的基本模块
        YOLO113n summary: 166 layers, 2,555,428 parameters, 0 gradients, 6.6 GFLOPs
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [00:11<00:00,  3.29it/s]
                       all       4952      12032      0.808      0.735      0.816      0.617
                 aeroplane        204        285      0.901      0.798      0.891       0.67
                   bicycle        239        337      0.885       0.84      0.912      0.704
                      bird        282        459      0.797      0.675       0.77      0.543
                      boat        172        263      0.749      0.677      0.752      0.495
                    bottle        212        469      0.847      0.569      0.706      0.482
                       bus        174        213      0.855      0.798      0.873      0.767
                       car        721       1201      0.883      0.826      0.914      0.738
                       cat        322        358      0.834      0.838       0.88      0.702
                     chair        417        756      0.747      0.496      0.638      0.433
                       cow        127        244      0.741      0.824      0.848       0.64
               diningtable        190        206      0.745      0.728      0.764       0.61
                       dog        418        489      0.786      0.753      0.849      0.663
                     horse        274        348      0.859      0.874      0.918      0.738
                 motorbike        222        325      0.876      0.801      0.899      0.656
                    person       2007       4528      0.891      0.764      0.874      0.607
               pottedplant        224        480      0.689       0.39       0.52       0.29
                     sheep         97        242        0.7       0.76      0.821      0.634
                      sofa        223        239      0.635      0.741      0.786      0.653
                     train        259        282      0.879      0.852      0.903      0.691
                 tvmonitor        229        308      0.871      0.699      0.796      0.615
    Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.2ms postprocess per image
    Results saved to runs/yolo11_VOC/113n133

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        middle_channels = int(in_channels * 0.5)

        final_conv_in_channels = 4 * middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.attn_W = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_W = nn.Sequential(
            Conv(
                c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True
            ),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )
        # self.conv_W = Conv(
        #     c1=middle_channels, c2=middle_channels, k=3, g=1, act=True
        # )

        self.attn_H = Conv(c1=middle_channels, c2=1, k=1, act=True)
        self.conv_H = nn.Sequential(
            Conv(
                c1=middle_channels, c2=middle_channels, k=3, g=middle_channels, act=True
            ),
            Conv(c1=middle_channels, c2=middle_channels, k=1, act=True),
        )
        # self.conv_H = Conv(
        #     c1=middle_channels, c2=middle_channels, k=3, g=1, act=True
        # )

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

        # 横向选择性聚合全局上下文信息
        logits_W = self.attn_W(x_main)
        context_scores_W = F.softmax(logits_W, dim=-1)
        x_plus_W = x_main + (x_main * context_scores_W).sum(-1, keepdim=True).expand_as(
            x_main
        )
        # 信息过滤
        x_plus_W = self.conv_W(x_plus_W)

        # 纵向选择性聚合全局左右上下文信息
        logits_H = self.attn_H(x_plus_W)
        context_scores_H = F.softmax(logits_H, dim=-2)
        x_plus_WH = x_plus_W + (x_plus_W * context_scores_H).sum(
            -2, keepdim=True
        ).expand_as(x_plus_W)
        # 信息过滤
        x_plus_WH = self.conv_H(x_plus_WH)

        # 多路径信息融合
        x_final = torch.cat((x_plus_WH, x_plus_W, x_main, x_short), dim=1)

        return self.final_conv(x_final) + residual_final


# class TransMoVE(nn.Module):
# """
# coco n scale
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.393
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.550
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.426
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.191
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.431
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.570
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.324
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.541
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.596
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.371
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.660
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.775
# Results saved to runs/yolo11_coco/113n
# """
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         num_experts: int = 9,
#         kernel_size: int = 3,
#     ):
#         super().__init__()

#         if out_channels == in_channels:
#             self.conv = nn.Identity()
#         else:
#             self.conv = Conv(in_channels, out_channels, k=1, act=True)

#         self.spatial_attn = DualAxisAggAttn(out_channels, out_channels)

#         self.norm1 = nn.GroupNorm(1, out_channels)
#         self.norm2 = nn.GroupNorm(1, out_channels)

#         self.local_extractor = MoVE_GhostModule(
#             out_channels, out_channels, kernel_size, num_experts
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         x = self.conv(x)

#         # 注意力子层
#         residual = x
#         norm_x = self.norm1(x)
#         x = residual + self.spatial_attn(norm_x)

#         # 前馈子层
#         residual = x
#         out = residual + self.local_extractor(x)

#         return out


# class TransMoVE_gamma(nn.Module):
#     """采用ELAN结构, 202504231537 相差0.3,效果不理想

#     """

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         num_experts: int = 9,
#         kernel_size: int = 3,
#     ):
#         super().__init__()

#         # 注意力子层
#         # --------------------------------------------------------------
#         self.spatial_attn = DualAxisAggAttn(out_channels, out_channels)

#         # ELAN结构
#         # ---------------------------------------------------------------
#         num_blocks = 2
#         num_in_block = 1
#         middle_ratio = 0.5

#         self.num_blocks = num_blocks

#         middle_channels = int(out_channels * middle_ratio)
#         block_channels = int(out_channels * middle_ratio)
#         self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
#         self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

#         self.blocks = nn.ModuleList()
#         for _ in range(num_blocks):
#             if num_in_block == 1:
#                 internal_block = MoVE_GhostModule(
#                     in_channels=middle_channels,
#                     out_channels=block_channels,
#                     num_experts=num_experts,
#                     kernel_size=kernel_size,
#                 )
#             else:
#                 internal_block = []
#                 for _ in range(num_in_block):
#                     internal_block.append(
#                         MoVE_GhostModule(
#                             in_channels=middle_channels,
#                             out_channels=block_channels,
#                             num_experts=num_experts,
#                             kernel_size=kernel_size,
#                         )
#                     )
#                 internal_block = nn.Sequential(*internal_block)

#             self.blocks.append(internal_block)

#         final_conv_in_channels = (
#             num_blocks * block_channels + int(out_channels * middle_ratio) * 2
#         )
#         self.final_conv = Conv(
#             c1=final_conv_in_channels, c2=out_channels, k=1, act=True
#         )

#         self.gamma_attn = nn.Parameter(torch.ones(out_channels), requires_grad=True)
#         self.gamma_elan = nn.Parameter(torch.ones(out_channels), requires_grad=True)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:


#         # 注意力子层
#         residual = x
#         x = residual + self.spatial_attn(x) * self.gamma_attn.view(-1, len(self.gamma_attn), 1, 1)

#         # ELAN子层
#         residual = x
#         x_short = self.short_conv(x)
#         x_main = self.main_conv(x)
#         block_outs = []
#         x_block = x_main
#         for block in self.blocks:
#             x_block = block(x_block)
#             block_outs.append(x_block)
#         x_final = torch.cat((*block_outs[::-1], x_main, x_short), dim=1)
#         return residual + self.final_conv(x_final) * self.gamma_elan.view(-1, len(self.gamma_elan), 1, 1)


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
                        # GhostModule(
                        #     in_channels=middle_channels,
                        #     out_channels=block_channels,
                        #     num_experts=num_experts,
                        #     kernel_size=kernel_size,
                        # )
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


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        multiple_of,
        ffn_dim_multiplier=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False, **factory_kwargs)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False, **factory_kwargs)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

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