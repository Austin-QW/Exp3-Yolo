from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

wandb.init(project="exp3",name = "exp3-yolo-no-aug-3")
# Load a pretrained model
model = YOLO("yolov8n.pt")
# Add Weights & Biases callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)
# Train the model on the custom dataset with augmentations
model.train(data="my_dataset_no_aug.yaml", epochs=5, imgsz=640, 
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
