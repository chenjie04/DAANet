# DualAxisAggAttn 模块复杂度分析

给定输入形状为 [B, C, H, W]，要求输出形状同样为 [B, C, H, W]，请详细分析 DualAxisAggAttn 类的空间复杂度和时间复杂度。

```python

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


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

        self.main_conv = Conv(c1=channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=channels, c2=middle_channels, k=1, act=True)

        self.qkv = nn.ModuleDict(
            {
                "W": nn.Conv2d(
                    in_channels=middle_channels,
                    out_channels=1 + 2 * middle_channels,
                    kernel_size=1,
                    bias=True,
                ),
                "H": nn.Conv2d(
                    in_channels=middle_channels,
                    out_channels=1 + 2 * middle_channels,
                    kernel_size=1,
                    bias=True,
                ),
            }
        )

        self.conv_fusion = nn.ModuleDict(
            {
                "W": light_ConvBlock(
                    in_channels=middle_channels, out_channels=middle_channels
                ),
                "H": light_ConvBlock(
                    in_channels=middle_channels, out_channels=middle_channels
                ),
            }
        )


        final_channels = int(2 * middle_channels)
        self.out_project = Conv(c1=final_channels, c2=channels, k=1, act=True)

    def _apply_axis_attention(self, x, axis):
        """通用轴注意力计算"""
        qkv = self.qkv[axis](x)
        query, key, value = torch.split(
            qkv, [1, self.middle_channels, self.middle_channels], dim=1
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
        x_W = self._apply_axis_attention(x_main, "W")
        x_W_fused = self.conv_fusion["W"](x_W) + x_W
        # 高轴注意力
        x_H = self._apply_axis_attention(x_W_fused, "H")
        x_H_fused = self.conv_fusion["H"](x_H) + x_H

        x_out = torch.cat([x_H_fused, x_short], dim=1)

        x_out = self.out_project(x_out)

        return x_out

```


我们来详细分析 `DualAxisAggAttn` 的参数复杂度（即权重数量）、**内存空间复杂度** 和  **时间复杂度（推理计算量）** ，假设输入张量 `x` 的形状为 `[B, C, H, W]`，设：

* `B`：batch size
* `C`：输入通道数
* `H, W`：空间尺寸
* `r`：`middle_ratio`，默认为 0.5
* `C_m = C * r`：中间特征通道数

---

### 🔧 一、参数复杂度

下面分别统计各层的参数量：

#### 1. `main_conv` 和 `short_conv`

都是 `1×1` 卷积，输入输出为 `C → C_m`：

* 每个卷积参数数：`C_m × C × 1 × 1 = C_m * C`
* 有两个这样的卷积，总参数量为：`2 × C_m × C = 2CrC`

#### 2. `qkv` 部分

对两个轴（W, H）各有一个卷积层：

* 每层输出通道为 `1 + 2*C_m`，输入为 `C_m`，kernel size = 1

  所以每层参数为：`(1 + 2C_m) × C_m × 1 × 1 = (1 + 2C_m) * C_m`
* 共有两个轴，共参数：`2 × (1 + 2C_m) * C_m`

#### 3. `light_ConvBlock`（两个轴各一个）

每个 `light_ConvBlock` 包含：

* 1x1 Conv（C_m → C_m）：`C_m × C_m`
* 3x3 DWConv（C_m → C_m，groups=C_m）：`3×3×C_m`
* 1x1 Conv（C_m → C_m）：`C_m × C_m`

总为：`2*C_m*C_m + 9*C_m`

两个轴总共：`2 × (2*C_m*C_m + 9*C_m)`

#### 4. `out_project`（1×1卷积）：输入通道为 `2*C_m`，输出通道为 `C`：

* 参数量为：`C × 2*C_m = 2CrC`

---

✅  **总参数复杂度** （忽略BN与bias）为：

$$
\text{Params} = 2CrC + 2(1 + 2CrC) + 4C_m^2 + 18C_m + 2CrC
$$

近似化后主导项是：

$$


$$

$$
O\mathcal(C^2r^2)
$$

---

$$


$$

### 🧠 二、空间复杂度（内存使用）

空间复杂度考虑以下几部分：

1. **中间特征图**
   * `x_main`、`x_short`：大小 `[B, C_m, H, W]`
   * 中间 `qkv` 输出：[B, 1+2C_m, H, W]（共两个轴）
   * 中间 `gate`、`context_vector`：各 [B, C_m, H, W]

这些都不重复释放，所以空间复杂度大约是：

$$
\mathcal{O}(B \cdot C_m \cdot H \cdot W)
$$

因为 `C_m = rC`，所以空间复杂度为：

$$


$$

$$
\mathcal{O}(B \cdot C \cdot H \cdot W)
$$


$$


$$

---

### ⏱ 三、时间复杂度（FLOPs）

主要考虑卷积和注意力计算：

#### 1. `Conv1x1` 和 `DWConv`

* `1×1` 卷积 FLOPs: `2 × H × W × in_channels × out_channels`
* `DWConv 3×3`: `2 × H × W × 3 × 3 × C_m`

#### 2. `qkv` 部分：

* `Conv1x1`: `2 × H × W × C_m × (1 + 2C_m)`，共两个轴

#### 3. `注意力操作`

* Softmax over H/W axis：代价小
* `key * context_scores`：点乘加权求和是：`B × C_m × H × W`

#### 4. 总体 FLOPs

总 FLOPs 主导项仍然是卷积，即：

* `light_ConvBlock` 中：DWConv、1x1Conv → `≈ 4 × H × W × C_m^2`
* `qkv` 中的 1x1 Conv → `≈ 2 × H × W × C_m^2`
* `out_project` → `2 × H × W × 2C_m × C = 4 × H × W × C_m × C`

所以时间复杂度约为：

$$
\mathcal(H \cdot W \cdot C^2 \cdot r^2)
$$

---

$$


$$

### ✅ 总结

| 项目                 | 复杂度                     | 主导项说明            |
| -------------------- | -------------------------- | --------------------- |
| **参数复杂度** | $\mathcal{O}(C^2 r^2)$   | 多个 `1×1`卷积为主 |
| **空间复杂度** | $\mathcal{O}(BCHW)$      | 中间张量与输入同阶    |
| **时间复杂度** | $\mathcal{O}(HWC^2 r^2)$ | 卷积计算为主          |

> 如果 `r=0.5`，那么相对普通 `C→C` 的 1×1 卷积大约计算量是 1/4～1/2 倍，代价可控。
>
