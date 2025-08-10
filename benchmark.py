from ultralytics.utils.benchmarks import benchmark

# Benchmark
benchmark(model="runs/yolo113_coco/113n2_41.7/weights/best.pt", data="coco8.yaml", imgsz=640, half=False, device="cpu")

"""
Benchmark on GPU
---------------------------------------------------------------------------------------------------------------------------
benchmark(model="runs/yolo11_VOC/n/weights/best.pt", data="VOC.yaml", imgsz=640, half=False, device="0")
Setup complete ✅ (32 CPUs, 125.5 GB RAM, 115.8/915.2 GB disk)

Benchmarks complete for runs/yolo113_VOC_ab/yolo11_n_60.77/weights/best.pt on VOC.yaml at imgsz=640 (988.92s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)     FPS
0                 PyTorch       ✅        5.3              0.6078                   2.26  443.05
1             TorchScript       ✅       10.4               0.605                   1.77  565.09
2                    ONNX       ✅       10.1              0.6049                   2.45  407.37
3                OpenVINO       ❌        0.0                   -                      -       -
4                TensorRT       ✅       14.0              0.6049                   1.46   683.8
5                  CoreML       ❌        0.0                   -                      -       -
6   TensorFlow SavedModel       ✅       25.5               0.605                  42.78   23.37
7     TensorFlow GraphDef       ✅       10.2               0.605                  44.76   22.34
8         TensorFlow Lite       ❌        0.0                   -                      -       -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -       -
10          TensorFlow.js       ❌        0.0                   -                      -       -
11           PaddlePaddle       ✅       20.2              0.6049                   5.45  183.38
12                    MNN       ✅       10.0              0.6049                  20.08    49.8
13                   NCNN       ✅       10.0               0.605                  27.47    36.4
14                    IMX       ❌        0.0                   -                      -       -
15                   RKNN       ❌        0.0                   -                      -       -
---------------------------------------------------------------------------------------------------------------------------
benchmark(model="runs/yolo113_VOC_ab/n_experts_4_63.17/weights/best.pt", data="VOC.yaml", imgsz=640, half=False, device="0")
Setup complete ✅ (32 CPUs, 125.5 GB RAM, 115.8/915.2 GB disk)

Benchmarks complete for runs/yolo113_VOC_ab/n_experts_4_63.17/weights/best.pt on VOC.yaml at imgsz=640 (1704.48s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)     FPS
0                 PyTorch       ✅        5.4              0.6317                   4.64  215.53
1             TorchScript       ✅       10.9              0.6283                   3.48  287.33
2                    ONNX       ✅       10.2              0.6283                    4.1  243.83
3                OpenVINO       ❌        0.0                   -                      -       -
4                TensorRT       ✅       16.2              0.6282                   2.38  420.46
5                  CoreML       ❌        0.0                   -                      -       -
6   TensorFlow SavedModel       ✅       30.2              0.6282                  87.44   11.44
7     TensorFlow GraphDef       ✅       16.1              0.6282                  91.46   10.93
8         TensorFlow Lite       ❌        0.0                   -                      -       -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -       -
10          TensorFlow.js       ❌        0.0                   -                      -       -
11           PaddlePaddle       ✅       20.6              0.6283                   7.84   127.5
12                    MNN       ✅       10.2              0.6283                  32.91   30.39
13                   NCNN       ✅       10.1              0.6282                  44.75   22.35
14                    IMX       ❌        0.0                   -                      -       -
15                   RKNN       ❌        0.0                   -                      -       -

---------------------------------------------------------------------------------------------------------------------------
# 这是self Attention
benchmark(model="runs/yolo113_VOC_ab/n_experts_4_self_attn_60.68/weights/best.pt", data="VOC.yaml", imgsz=640, half=False, device="0")
Setup complete ✅ (32 CPUs, 125.5 GB RAM, 115.8/915.2 GB disk)

Benchmarks complete for runs/yolo113_VOC_ab/n_experts_4_self_attn_60.68/weights/best.pt on VOC.yaml at imgsz=640 (923.75s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)     FPS
0                 PyTorch       ✅        5.3              0.6066                   4.68  213.55
1             TorchScript       ❌        0.0                   -                      -       -
2                    ONNX       ✅       14.6              0.5341                   2.81  355.77
3                OpenVINO       ❌        0.0                   -                      -       -
4                TensorRT       ✅       19.2               0.534                   1.77  565.64
5                  CoreML       ❌        0.0                   -                      -       -
6   TensorFlow SavedModel       ✅       36.8               0.534                  53.17   18.81
7     TensorFlow GraphDef       ✅       14.7               0.534                  53.72   18.62
8         TensorFlow Lite       ❌        0.0                   -                      -       -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -       -
10          TensorFlow.js       ❌        0.0                   -                      -       -
11           PaddlePaddle       ❌        0.0                   -                      -       -
12                    MNN       ✅       14.6              0.0001                  26.85   37.24
13                   NCNN       ❌        0.0                   -                      -       -
14                    IMX       ❌        0.0                   -                      -       -
15                   RKNN       ❌        0.0                   -                      -       -   




---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
Benchmark on CPU 
---------------------------------------------------------------------------------------------------------------------------
benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False, device="cpu")
Setup complete ✅ (32 CPUs, 125.5 GB RAM, 116.0/915.2 GB disk)

Benchmarks complete for yolo11n.pt on coco8.yaml at imgsz=640 (126.70s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)    FPS
0                 PyTorch       ✅        5.4                0.61                  20.94  47.76
1             TorchScript       ✅       10.5              0.6082                  36.89   27.1
2                    ONNX       ✅       10.2              0.6082                  20.08  49.81
3                OpenVINO       ✅       10.4              0.6082                  19.71  50.72
4                TensorRT       ❌        0.0                   -                      -      -
5                  CoreML       ❎        5.2                   -                      -      -
6   TensorFlow SavedModel       ✅       25.8              0.6082                  77.64  12.88
7     TensorFlow GraphDef       ✅       10.3              0.6082                  81.51  12.27
8         TensorFlow Lite       ✅       10.3              0.6082                  52.99  18.87
9     TensorFlow Edge TPU       ❌        0.0                   -                      -      -
10          TensorFlow.js       ❎       10.5                   -                      -      -
11           PaddlePaddle       ✅       20.4              0.6082                 105.65   9.47
12                    MNN       ✅       10.1              0.6082                  23.06  43.37
13                   NCNN       ✅       10.2              0.6082                  36.76  27.21
14                    IMX       ❌        0.0                   -                      -      -
15                   RKNN       ❌        0.0                   -                      -      -

benchmark(model="runs/yolo113_coco/113n2_41.7/weights/best.pt", data="coco8.yaml", imgsz=640, half=False, device="cpu")
Setup complete ✅ (32 CPUs, 125.5 GB RAM, 116.0/915.2 GB disk)

Benchmarks complete for runs/yolo113_coco/113n2_41.7/weights/best.pt on coco8.yaml at imgsz=640 (301.14s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)    FPS
0                 PyTorch       ✅        5.5              0.6975                  31.13  32.12
1             TorchScript       ✅       11.0               0.694                  68.06  14.69
2                    ONNX       ✅       10.3               0.694                   24.8  40.33
3                OpenVINO       ✅       10.6               0.694                   26.1  38.32
4                TensorRT       ❌        0.0                   -                      -      -
5                  CoreML       ❎        5.3                   -                      -      -
6   TensorFlow SavedModel       ✅       30.5               0.694                 132.21   7.56
7     TensorFlow GraphDef       ✅       16.3               0.694                 134.45   7.44
8         TensorFlow Lite       ✅       13.3               0.694                   64.7  15.46
9     TensorFlow Edge TPU       ❌        0.0                   -                      -      -
10          TensorFlow.js       ❎       12.3                   -                      -      -
11           PaddlePaddle       ✅       20.9               0.694                 124.91   8.01
12                    MNN       ✅       10.3               0.694                   33.8  29.59
13                   NCNN       ✅       10.2               0.694                  45.75  21.86
14                    IMX       ❌        0.0                   -                      -      -
15                   RKNN       ❌        0.0                   -                      -      -

"""