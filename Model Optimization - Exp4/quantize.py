# 方案二：量化
from ultralytics import YOLO

model = YOLO("/workspace/Exp3/runs/detect/aug/weights/best.pt")
model.export(
    format="engine",
    dynamic=True,  
    batch=3,  
    workspace=4,  
    #half=True,
    int8=True,
    data="/workspace/Exp3/my_dataset.yaml",  
)

# Load the exported TensorRT INT8 model 注意这里我手动改了engine的名字，不然生成的两个模型都是best.engine
# model = YOLO("/workspace/Exp3/runs/detect/aug/weights/best_FP16.engine", task="detect")
model = YOLO("/workspace/Exp3/runs/detect/aug/weights/best_INT8.engine", task="detect")

# Run inference
results = model.val(data="/workspace/Exp3/my_dataset.yaml")
print(f"mAP50-95:{results.box.map}")