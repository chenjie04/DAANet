from ultralytics.utils.benchmarks import benchmark

# Benchmark
benchmark(model="runs/yolo113_VOC_ab/n_experts_16_learnable_weights_62.76/weights/best.pt", data="VOC.yaml", imgsz=640, half=False, device="0")

"""
Benchmark on GPU
benchmark(model="runs/yolo11_VOC/n/weights/best.pt", data="VOC.yaml", imgsz=640, half=False, device="0")
Benchmarks complete for runs/yolo11_VOC/n/weights/best.pt on VOC.yaml at imgsz=640 (1076.08s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)     FPS
0                 PyTorch       ✅        5.3              0.6078                   2.29  436.95
1             TorchScript       ✅       10.4               0.605                   1.77  565.83
2                    ONNX       ✅       10.1               0.605                  15.56   64.28
3                OpenVINO       ❌        0.0                   -                      -       -
4                TensorRT       ✅       13.8              0.6049                   1.43  696.68
5                  CoreML       ❌        0.0                   -                      -       -
6   TensorFlow SavedModel       ✅       25.5               0.605                  46.31   21.59
7     TensorFlow GraphDef       ✅       10.2               0.605                  46.52    21.5
8         TensorFlow Lite       ❌        0.0                   -                      -       -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -       -
10          TensorFlow.js       ❌        0.0                   -                      -       -
11           PaddlePaddle       ✅       20.2              0.6049                   5.43  184.06
12                    MNN       ✅       10.0              0.6049                  20.19   49.52
13                   NCNN       ✅       10.0               0.605                  27.62    36.2
14                    IMX       ❌        0.0                   -                      -       -
15                   RKNN       ❌        0.0                   -                      -       -

benchmark(model="runs/yolo11_voc_attn_ab/113n_experts_1624_61.9/weights/best.pt", data="VOC.yaml", imgsz=640, half=False, device="0")
Benchmarks complete for runs/yolo11_voc_attn_ab/113n_experts_1624_61.9/weights/best.pt on VOC.yaml at imgsz=640 (2136.98s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)     FPS
0                 PyTorch       ✅        5.5              0.6194                   5.21  191.85
1             TorchScript       ✅       10.8              0.6155                   4.53  220.86
2                    ONNX       ✅       10.4              0.6155                  31.18   32.07 
3                OpenVINO       ❌        0.0                   -                      -       -
4                TensorRT       ✅       20.1              0.6154                   2.92   341.9
5                  CoreML       ❌        0.0                   -                      -       -
6   TensorFlow SavedModel       ✅       30.6              0.6155                  99.94   10.01
7     TensorFlow GraphDef       ✅       16.3              0.6155                  98.56   10.15
8         TensorFlow Lite       ❌        0.0                   -                      -       -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -       -
10          TensorFlow.js       ❌        0.0                   -                      -       -
11           PaddlePaddle       ❌        0.0                   -                      -       -
12                    MNN       ✅       10.3              0.6154                  39.54   25.29
13                   NCNN       ✅       10.2              0.6155                  66.05   15.14
14                    IMX       ❌        0.0                   -                      -       -
15                   RKNN       ❌        0.0                   -                      -       -

Benchmarks complete for runs/yolo11_voc_attn_ab/113n_experts_1624_61.9/weights/best.pt on VOC.yaml at imgsz=640 (2023.19s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)     FPS
0                 PyTorch       ✅        5.5              0.6194                   5.21  191.96
1             TorchScript       ✅       10.8              0.6155                   4.53  220.62
2                    ONNX       ✅       10.4              0.6155                   5.38  185.79
3                OpenVINO       ❌        0.0                   -                      -       -
4                TensorRT       ✅       19.5              0.6154                   2.93  341.68
5                  CoreML       ❌        0.0                   -                      -       -
6   TensorFlow SavedModel       ✅       30.6              0.6155                 101.59    9.84
7     TensorFlow GraphDef       ✅       16.3              0.6155                 101.66    9.84
8         TensorFlow Lite       ❌        0.0                   -                      -       -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -       -
10          TensorFlow.js       ❌        0.0                   -                      -       -
11           PaddlePaddle       ❌        0.0                   -                      -       -
12                    MNN       ✅       10.3              0.6154                  39.48   25.33
13                   NCNN       ✅       10.2              0.6155                   65.8    15.2
14                    IMX       ❌        0.0                   -                      -       -
15                   RKNN       ❌        0.0                   -                      -       -

# 这是self Attention
Benchmarks complete for runs/yolo11_voc_ab/113n_only_self_attn12/weights/best.pt on VOC.yaml at imgsz=640 (1160.84s)                                                                                              
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed                                                                                                                    
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)     FPS                                                                                                                  
0                 PyTorch       ✅        5.4               0.608                   4.89  204.42                                                                                                                  
1             TorchScript       ❌        0.0                   -                      -       -                                                                                                                  
2                    ONNX       ✅       14.8              0.5245                   4.17  239.86
3                OpenVINO       ❌        0.0                   -                      -       -
4                TensorRT       ✅       22.9              0.5246                   2.32  430.98
5                  CoreML       ❌        0.0                   -                      -       -
6   TensorFlow SavedModel       ✅       37.2              0.5245                  67.41   14.83
7     TensorFlow GraphDef       ✅       14.9              0.5245                  69.22   14.45
8         TensorFlow Lite       ❌        0.0                   -                      -       -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -       -
10          TensorFlow.js       ❌        0.0                   -                      -       -
11           PaddlePaddle       ❌        0.0                   -                      -       -         
12                    MNN       ✅       14.8              0.5246                  32.98   30.32         
13                   NCNN       ❌        0.0                   -                      -       -
14                    IMX       ❌        0.0                   -                      -       -                                                                                                                  
15                   RKNN       ❌        0.0                   -                      -       -

Benchmarks complete for runs/yolo113_DUO_v2/n2/weights/best.pt on DUO.yaml at imgsz=640 (538.35s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)     FPS
0                 PyTorch       ✅        5.5              0.6491                   3.64  275.03
1             TorchScript       ✅       10.9              0.6442                   3.59  278.53
2                    ONNX       ✅       10.5              0.6441                   4.25  235.26
3                OpenVINO       ❌        0.0                   -                      -       -
4                TensorRT       ✅       17.4              0.6441                   2.46  406.72
5                  CoreML       ❌        0.0                   -                      -       -
6   TensorFlow SavedModel       ✅       26.6              0.6441                  68.18   14.67
7     TensorFlow GraphDef       ✅       10.6              0.6441                  68.23   14.66
8         TensorFlow Lite       ❌        0.0                   -                      -       -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -       -
10          TensorFlow.js       ❌        0.0                   -                      -       -
11           PaddlePaddle       ❌        0.0                   -                      -       -
12                    MNN       ✅       10.5              0.6442                  41.19   24.28
13                   NCNN       ✅       10.4              0.6441                  64.48   15.51
14                    IMX       ❌        0.0                   -                      -       -
15                   RKNN       ❌        0.0                   -                      -       -
----------------------------------------------------------------------------------------------------------------------------------------------
Benchmark on CPU
benchmark(model="runs/yolo11_VOC/n/weights/best.pt", data="VOC.yaml", imgsz=640, half=False, device="cpu")
Benchmarks complete for runs/yolo11_VOC/n/weights/best.pt on VOC.yaml at imgsz=640 (79751.00s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)    FPS
0                 PyTorch       ✅        5.3              0.6077                  17.45  57.29
1             TorchScript       ✅       10.4               0.605                  18.04  55.44
2                    ONNX       ✅       10.1               0.605                  15.55  64.33
3                OpenVINO       ✅       10.3               0.605                   17.3  57.79
4                TensorRT       ❌        0.0                   -                      -      -
5                  CoreML       ❎        5.1                   -                      -      -
6   TensorFlow SavedModel       ✅       25.5               0.605                  33.31  30.02
7     TensorFlow GraphDef       ✅       10.2               0.605                  33.04  30.26
8         TensorFlow Lite       ✅       10.2               0.605                  49.92  20.03
9     TensorFlow Edge TPU       ❌        0.0                   -                      -      -
10          TensorFlow.js       ❎       10.4                   -                      -      -
11           PaddlePaddle       ✅       20.2               0.605                  78.36  12.76
12                    MNN       ✅       10.0              0.6049                  21.11  47.37
13                   NCNN       ✅       10.0               0.605                  27.26  36.68
14                    IMX       ❌        0.0                   -                      -      -
15                   RKNN       ❌        0.0                   -                      -      -

Benchmarks complete for runs/yolo11_voc_attn_ab/113n_experts_1624_61.9/weights/best.pt on VOC.yaml at imgsz=640 (59839.37s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)    FPS
0                 PyTorch       ✅        5.5              0.6194                  29.86  33.48
1             TorchScript       ✅       10.8              0.6155                  28.57   35.0
2                    ONNX       ✅       10.4              0.6155                  28.42  35.19
3                OpenVINO       ✅       10.7              0.6155                  31.61  31.64
4                TensorRT       ❌        0.0                   -                      -      -
5                  CoreML       ❎        5.4                   -                      -      -
6   TensorFlow SavedModel       ✅       30.6              0.6155                  80.82  12.37
7     TensorFlow GraphDef       ✅       16.3              0.6155                   82.8  12.08
8         TensorFlow Lite       ✅       13.3              0.6155                  86.26  11.59
9     TensorFlow Edge TPU       ❌        0.0                   -                      -      -
10          TensorFlow.js       ❎       16.6                   -                      -      -
11           PaddlePaddle       ❌        0.0                   -                      -      -
12                    MNN       ✅       10.3              0.6154                  56.25  17.78
13                   NCNN       ✅       10.2              0.6155                 106.71   9.37
14                    IMX       ❌        0.0                   -                      -      -
15                   RKNN       ❌        0.0                   -                      -      -

"""