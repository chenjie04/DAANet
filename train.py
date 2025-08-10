# import os
# os.environ['MKL_SERVICE_FORCE_INTEL'] = 'GNU'


from ultralytics import YOLO, data

# Load a model
# model = YOLO("yolo113n.yaml")
model = YOLO("runs/yolo113_coco/113s2_47.51/weights/best.pt")

# Train the model
train_results = model.train(
    # resume=True,
    # data="VOC.yaml",  # path to dataset YAML
    # data="DUO.yaml",
    # data="Brackish.yaml",
    # data="TrashCAN_material.yaml",
    data="coco.yaml",
    epochs=500,  # number of training epochs
    batch=64,
    imgsz=640,  # training image size
    scale=0.9,  # N:0.5, S:0.9; M:0.9; L:0.9; X:0.9
    mosaic=1.0,
    mixup=0.05,  # N:0.0, S:0.05; M:0.15; L:0.15; X:0.2
    copy_paste=0.15,  # N:0.1,S:0.15; M:0.4; L:0.5; X:0.6
    device=[0, 1],  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    project="training_log/yolo113_coco",
    name="n_4e"
)
