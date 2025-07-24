# import os
# os.environ['MKL_SERVICE_FORCE_INTEL'] = 'GNU'


from ultralytics import YOLO, data

# Load a model
model = YOLO("yolo114n.yaml")
# model = YOLO("training_log/yolo11_coco/m2/weights/last.pt")

# Train the model
train_results = model.train(
    # resume=True,
    data="VOC.yaml",  # path to dataset YAML
    # data="DUO.yaml",
    # data="Brackish.yaml",
    # data="TrashCAN_material.yaml",
    # data="coco.yaml",
    epochs=500,  # number of training epochs
    batch=32,
    imgsz=640,  # training image size
    device=[0, 1],  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    project="training_log/yolo114_VOC",
    name="n"
)
