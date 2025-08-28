# DA<sup>3</sup>Net: Dual-Axis Aggregation Attention Network for Efficient Object Detection

This project is built based on [Ultralytics](https://github.com/ultralytics/ultralytics).

Self-attention has demonstrated great potential for improving performance in object detection. However, the quadratic computational complexity with respect to the input size hinders its feasibility for high-resolution vision tasks in real-time scenarios. To effectively address this issue, we reinterpret self-attention as selective global context aggregation and propose a Dual-Axis Aggregation Attention (DA<sup>3</sup>) module. It achieves global perception with linear complexity by sequentially aggregating features along the distinct spatial axes, followed by depth-wise separable convolution to enhance feature selection. Based on this mechanism, we develop DA<sup>3</sup>Net, a lightweight backbone for efficient object detection. It also incorporates an enhanced local feature extraction module to improve the representation learning of fine-grained visual details guided by inductive bias compensation, which is an essential complement to long-range dependencies.

Extensive experiments validate the effectiveness of the proposed approach. On MS COCO, DA<sup>3</sup>Net-N achieves 41.7 AP, exceeding YOLO11-N by 2.2 AP. Ablation studies on the PASCAL VOC further demonstrate the effectiveness and efficiency of each design component, with DA<sup>3</sup>Net-N attaining 63.17 mAP, which is 2.4 mAP higher than YOLO11-N. On a RTX A5000 GPU, it runs at 420.63 FPS in TensorRT format—over 12x faster than standard self-attention—demonstrating strong real-time capability. Moreover, generalization experiments on underwater object detection further validate the robustness and adaptability of the proposed approach.

![](images/graph_abstract.png)

## Results