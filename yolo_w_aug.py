from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo11n.pt")

# Train the model on the custom dataset
model.train(data="my_dataset.yaml", epochs=70, imgsz=640)