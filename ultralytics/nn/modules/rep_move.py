import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

from typing import Tuple

# 效果极差

class Rep_MoVE(nn.Module):
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

        self.infer_mode = False

        self.num_experts = num_experts
        self.in_channels = in_channels

        if self.infer_mode:
            self.reparam_expert = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    groups=in_channels,
                    bias=False,
                ),
                nn.InstanceNorm2d(in_channels),
                nn.SiLU(),
            )
        else:
            experts = list()
            for i in range(num_experts):
                expert = self._conv_in(
                        in_channels,
                        in_channels,
                        kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                        groups=in_channels,
                        bias=False,
                    )
                # 在这里添加激活之后，就无法保证重参数化等价,但是这里进行非线性激活对增强模型能力很重要
                expert.add_module("act", nn.SiLU()) 
                experts.append(
                    expert
                )
            self.experts = nn.ModuleList(experts)

            # 初始化缩放因子为1.0（保持初始输出不变）
            self.scales = nn.ParameterList(
                [
                    nn.Parameter(torch.ones(in_channels))
                    for _ in range(num_experts)
                ]
            )

            self.act = nn.SiLU() # 用于重参数化之后的激活

        self.out_prj = Conv(
            c1=in_channels,
            c2=out_channels,
            k=1,
            act=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.infer_mode:
            # print("Infer mode")
            x = self.reparam_expert(x)
            x = self.out_prj(x)
            return x
        else:
            # print("Train mode")
            moe_out = 0
            for i in range(self.num_experts):
                moe_out += self.experts[i](x) * self.scales[i].view(1, -1, 1, 1)

            moe_out = self.out_prj(moe_out)

            return moe_out

    def _conv_in(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        bias=False,
    ):
        """Helper method to construct conv-instancenorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-IN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
        )
        mod_list.add_module(
            "norm",
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
        )
        return mod_list

    def reparameterize(self):
        if hasattr(self, "reparam_expert"):
            return

        self.reparam_expert = nn.Conv2d(
            in_channels=self.experts[0].conv.in_channels,
            out_channels=self.experts[0].conv.out_channels,
            kernel_size=self.experts[0].conv.kernel_size,
            stride=self.experts[0].conv.stride,
            padding=self.experts[0].conv.padding,
            groups=self.experts[0].conv.groups,
            bias=False,
        )
        self.expert_norm = nn.InstanceNorm2d(self.experts[0].conv.out_channels)

        kernel_conv = 0
        bias_conv = 0
        for i in range(self.num_experts):
            _kernel = self.experts[i].conv.weight
            running_mean = self.experts[i].norm.running_mean
            running_var = self.experts[i].norm.running_var
            gamma = self.experts[i].norm.weight
            beta = self.experts[i].norm.bias
            eps = self.experts[i].norm.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            _kernel = _kernel * t
            _beta = beta - running_mean * gamma / std
            kernel_conv += _kernel * self.scales[i]
            bias_conv += _beta * self.scales[i]

        self.reparam_expert.weight.data = kernel_conv

        self.reparam_expert = nn.Sequential(
            self.reparam_expert,
            self.act,
        )

        delattr(self, "experts")
        delattr(self, "scales")

        self.infer_mode = True


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = Rep_MoVE(3, 64)
    # print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.eval()
    y = model(x)
    print(y.shape)

    model.reparameterize()
    # print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    with torch.inference_mode():
        y1 = model(x)

    print(torch.allclose(y, y1), torch.norm(y - y1))
    # 使用容差比较结果
    is_close = torch.allclose(y, y1, atol=1e-5, rtol=1e-3)
    max_diff = (y - y1).abs().max().item()

    print(f"输出是否一致: {is_close}")
    print(f"最大差异: {max_diff:.6f}")
