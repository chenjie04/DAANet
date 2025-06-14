import time
import torch
from ultralytics import YOLO
from collections import defaultdict

class LayerTimer:
    def __init__(self):
        self.times = defaultdict(list)
    
    def __call__(self, module, input, output):
        torch.cuda.synchronize()  # 同步GPU操作
        elapsed = time.time() - self.start_time
        self.times[module.__class__.__name__].append(elapsed)

# 注册钩子
timer = LayerTimer()
model = YOLO('yolo113n.yaml').to('cuda')
for name, layer in model.named_modules():
    layer.register_forward_hook(timer)

# 运行推理
input_tensor = torch.randn(1, 3, 640, 640).to('cuda')
timer.start_time = time.time()
model(input_tensor)

# 输出各层平均耗时
for name, times in timer.times.items():
    # avg_time = sum(times) / len(times) * 1000  # 毫秒
    # print(f"{name}: {avg_time:.2f} ms")
    total_time = sum(times) * 1000
    num = len(times)
    print(f"{name}: {num} times and {total_time:.2f} ms")