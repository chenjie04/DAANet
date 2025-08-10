from ultralytics import YOLO

# model = YOLO("yolov8n.pt")
model = YOLO(
    "runs/yolo113_VOC_ab/n_experts_4_63.17/weights/best.pt"
)  # 模型文件路径

# print(model)
# Perform inference on an image
results = model(
    "runs/feature_visualization/002434.jpg", visualize=True, project="runs/feature_visualization", name="n_experts_4_63.17"
)  # 要预测图片路径和使用可视化
