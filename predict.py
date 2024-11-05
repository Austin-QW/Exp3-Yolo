from ultralytics import YOLO
import os

# 设置测试集路径
yolo_test_images = '/workspace/Exp3/dataset/test/images'
obb_test_images = '/workspace/Exp3/obb_dataset/test/images'

# 设置结果保存路径
output_dir = "/workspace/Exp3/predictions"
os.makedirs(output_dir, exist_ok=True)

# 加载每个模型权重
models = {
    "no_augmentation": YOLO("/workspace/Exp3/runs/detect/no_aug/weights/best.pt"),
    "augmentation": YOLO("/workspace/Exp3/runs/detect/aug/weights/best.pt"),
    "yolo_obb": YOLO("/workspace/Exp3/runs/obb/aug/weights/best.pt"),
    "yolo_obb_no_aug": YOLO("/workspace/Exp3/runs/obb/no_aug/weights/best.pt")
}

# 设置不同模型对应的测试集
test_datasets = {
    "no_augmentation": yolo_test_images,
    "augmentation": yolo_test_images,
    "yolo_obb": obb_test_images,
    "yolo_obb_no_aug": obb_test_images
}

# 对测试集进行预测并保存结果
for model_name, model in models.items():
    # 获取当前模型的测试集路径
    test_images = test_datasets[model_name]
    
    # 运行预测，结果保存到对应的目录
    results = model.predict(source=test_images, save=True, save_txt=True, save_conf=True,
                            project=os.path.join(output_dir, model_name), exist_ok=True)
    print(f"{model_name} 模型的预测完成，结果保存在 {os.path.join(output_dir, model_name)}")
