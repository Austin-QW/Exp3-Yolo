from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

# Load a pretrained model
model = YOLO("yolo11n-obb.pt")

# Add W&B callback for Ultralytics see https://docs.wandb.ai/guides/integrations/ultralytics/
add_wandb_callback(model, enable_model_checkpointing=True)

# Train the model on the custom dataset
model.train(project="exp3-yolo-obb",data="my_dataset_obb.yaml", epochs=100, imgsz=640) # 针对obb要修改数据标注格式

# Finish the W&B run
wandb.finish()