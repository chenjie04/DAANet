from ultralytics import YOLO

#model = YOLO("yolov8n.pt")
model = YOLO("runs/yolo11_voc_attn_ab/113n_experts_1624_61.9/weights/best.pt")  # 模型文件路径

results = model("D:/yolov8/1.png", visualize=True)  # 要预测图片路径和使用可视化