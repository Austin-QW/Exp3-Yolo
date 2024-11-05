from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolo11n-obb.pt")

# 在自定义数据集上训练模型，禁用数据增强
model.train(
    data="my_dataset_obb_no_aug.yaml",
    epochs=70,
    imgsz=640,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,
    mosaic=0.0,
    mixup=0.0,
    auto_augment=None,
    erasing=0.0
)
