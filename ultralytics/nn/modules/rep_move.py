import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

from typing import Tuple


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
                nn.BatchNorm2d(in_channels),
                nn.SiLU(),
            )
        else:
            experts = list()
            for i in range(num_experts):
                experts.append(
                    self._conv_bn(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                        groups=in_channels,
                        bias=False,
                    )
                )
            self.experts = nn.ModuleList(experts)
            self.act = nn.SiLU()

            # 初始化缩放因子为1.0（保持初始输出不变）
            self.scales = nn.ParameterList(
                [nn.Parameter(torch.ones(in_channels) * 0.1) for _ in range(num_experts)]
            )

        self.out_prj = self._conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
        )
        self.out_prj.add_module("silu", nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.infer_mode:
            print("Infer mode")
            x = self.reparam_expert(x)
            x = self.out_prj(x)
            return x
        else:
            print("Train mode")
            moe_out = 0
            for i in range(self.num_experts):
                moe_out += self.experts[i](x) * self.scales[i].view(1, -1, 1, 1)
            moe_out = self.act(moe_out)

            moe_out = self.out_prj(moe_out)

            return moe_out

    def _conv_bn(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        bias=False,
    ):
        """Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
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
        mod_list.add_module("bn", nn.BatchNorm2d(out_channels))
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
            bias=True,
        )

        kernel_conv = 0
        bias_conv = 0
        for i in range(self.num_experts):
            _kernel, _bias = self._fuse_bn_tensor(self.experts[i])
            kernel_conv += _kernel * self.scales[i]
            bias_conv += _bias

        self.reparam_expert.weight.data = kernel_conv
        self.reparam_expert.bias.data = bias_conv

        self.reparam_expert = nn.Sequential(
            self.reparam_expert,
            self.act,
        )

        delattr(self, "experts")
        delattr(self, "scales")

        self.infer_mode = True

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[
                        i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                    ] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


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
