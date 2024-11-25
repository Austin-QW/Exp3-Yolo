from ultralytics import YOLO
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import onnx
from onnxconverter_common import float16

# 加载预训练的 YOLO 模型
model = YOLO("/workspace/Exp3/runs/detect/aug/weights/best.pt")
#最佳模型性能计算
results = model.val(data="/workspace/Exp3/my_dataset.yaml")
print(f"mAP50-95:{results.box.map}")


# 方案一：剪枝
def prune_model(model, amount=0.05):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d): # 检测module是不是conv2d类
            prune.l1_unstructured(module, name='weight', amount=amount) #L1无结构剪枝，移除权重最小的5%。
            prune.remove(module, 'weight') 
    return model

pytorch_model = model.model
pruned_torch_model = prune_model(pytorch_model, amount=0.05) # 剪枝 5% 的权重
model.model = pruned_torch_model
model.save( "pruned_model.pt")
print("pruned model saved")

#剪枝后性能计算
pruned_model = YOLO("pruned_model.pt")
results = pruned_model.val(data="/workspace/Exp3/my_dataset.yaml")
print(f"mAP50-95:{results.box.map}")

# 微调剪枝后的模型
results = model.train(data = "/workspace/Exp3/my_dataset.yaml", epochs= 50)
model.save("pruned_finetuned_model.pt")
print("pruned_finetuned_model saved")

# 微调剪枝后性能计算
pruned_finetuned_model = YOLO("pruned_finetuned_model.pt")
results = pruned_finetuned_model.val(data="/workspace/Exp3/my_dataset.yaml")
print(f"mAP50-95:{results.box.map}")




