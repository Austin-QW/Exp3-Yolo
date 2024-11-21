import os
import time
from ultralytics import YOLO
import json

# 定义每个模型的 YAML 文件和测试集图片路径
datasets = {
    "no_augmentation": {
        "yaml": "/workspace/Exp3/my_dataset_no_aug.yaml",
        "images": "/workspace/Exp3/dataset/test/images"
    },
    "augmentation": {
        "yaml": "/workspace/Exp3/my_dataset.yaml",
        "images": "/workspace/Exp3/dataset/test/images"
    },
    "yolo_obb": {
        "yaml": "/workspace/Exp3/my_dataset_obb.yaml",
        "images": "/workspace/Exp3/obb_dataset/test/images"
    },
    "yolo_obb_no_aug": {
        "yaml": "/workspace/Exp3/my_dataset_obb_no_aug.yaml",
        "images": "/workspace/Exp3/obb_dataset/test/images"
    }
}

# 定义模型路径
models = {
    "no_augmentation": YOLO("/workspace/Exp3/runs/detect/no_aug/weights/best.pt"),
    "augmentation": YOLO("/workspace/Exp3/runs/detect/aug/weights/best.pt"),
    "yolo_obb": YOLO("/workspace/Exp3/runs/obb/aug/weights/best.pt"),
    "yolo_obb_no_aug": YOLO("/workspace/Exp3/runs/obb/no_aug/weights/best.pt")
}

# 存储结果
results = {}

# 遍历每个模型和对应的数据集
for model_name, model in models.items():
    dataset_info = datasets[model_name]
    yaml_path = dataset_info["yaml"]
    images_path = dataset_info["images"]
    print(f"Evaluating {model_name} on dataset {yaml_path}...")

    # 测试集推理并统计时间
    start_time = time.time()
    predictions = model.predict(source=images_path, save=False)  # 输入测试集路径进行推理
    end_time = time.time()

    # 计算 FPS
    total_images = len(predictions)  # 推理的图片总数
    total_time = end_time - start_time  # 推理总时间
    fps = total_images / total_time if total_time > 0 else 0

    # 模型验证
    metrics = model.val(data=yaml_path)  # 使用 YAML 文件进行验证
    det_metrics = metrics.box  # 获取检测指标的计算对象

    # 提取性能指标
    precision, recall, map50, map50_95 = det_metrics.mean_results()  # 提取主要性能指标
    fps = total_images / total_time if total_time > 0 else 0  # FPS 计算

    # 保存结果
    results[model_name] = {
        "precision": precision,
        "recall": recall,
        "mAP@50": map50,
        "mAP@50-95": map50_95,
        "FPS": fps,
    }

    print(f"{model_name}: P = {precision:.4f}, R = {recall:.4f}, mAP@50 = {map50:.4f}, mAP@50-95 = {map50_95:.4f}, FPS = {fps:.2f}")


# 将结果保存为 JSON 文件
output_file = "/workspace/Exp3/predictions/evaluation_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")
