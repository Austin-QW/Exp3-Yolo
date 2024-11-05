from ultralytics import YOLO

model = YOLO("yolo11n-obb.pt")

model.train(data="my_dataset_obb.yaml", epochs=70, imgsz=640)
