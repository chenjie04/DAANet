import torch
from ultralytics import YOLO, NAS
from thop import profile


model = YOLO("yolo113n.yaml")
# model = YOLO("runs/yolo113_VOC_ab/n_experts_4_self_attn_60.68/weights/best.pt")
# print(model)

model.info(detailed=False)

# Load a COCO-pretrained YOLO-NAS-s model
# model = NAS("yolo_nas_s.pt")

# # Display model information (optional)
# model.info()