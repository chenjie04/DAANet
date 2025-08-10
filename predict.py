from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("runs/yolo113_Brackish/s_experts_4_88.50/weights/best.pt")
model.info()

# Run inference on 'bus.jpg' with arguments
model.predict("underwater_result_visualization/brackish/2019-03-20_20-08-39to2019-03-20_20-08-48_1-0076.png", save=True, imgsz=640, conf=0.5)