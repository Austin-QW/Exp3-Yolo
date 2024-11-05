import os
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# 各模型对应的真实标签路径
ground_truth_dirs = {
    "no_augmentation": '/workspace/Exp3/dataset/test/labels',
    "augmentation": '/workspace/Exp3/dataset/test/labels',
}

# 各模型预测结果的目录
predictions_dir = '/workspace/Exp3/predictions'

# 初始化 mAP 计算器
map_calculator = MeanAveragePrecision()

# 函数：加载标签文件，根据模型类型选择适当的格式
def load_labels(label_path, model_type, is_prediction=False):
    boxes, labels, scores = [], [], []

    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            
            if model_type in ["no_augmentation", "augmentation"]:
                # YOLO 格式：class_id x_center y_center width height -> 转换为 [xmin, ymin, xmax, ymax]
                x_center, y_center, width, height = map(float, parts[1:5])
                xmin = x_center - width / 2
                ymin = y_center - height / 2
                xmax = x_center + width / 2
                ymax = y_center + height / 2
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id)
                
                # 如果是预测结果，读取置信度分数
                if is_prediction:
                    scores.append(float(parts[-1]))  # 置信度分数

    return boxes, labels, scores

# 对每个 YOLO 模型的预测结果和真实标签进行 mAP 计算
metrics_results = {}

for model_name in ["no_augmentation", "augmentation"]:
    gt_dir = ground_truth_dirs[model_name]
    pred_dir = os.path.join(predictions_dir, model_name, "predict", "labels")

    for file_name in os.listdir(gt_dir):
        gt_file = os.path.join(gt_dir, file_name)
        pred_file = os.path.join(pred_dir, file_name)

        # 检查预测文件是否存在
        if not os.path.exists(pred_file):
            print(f"Warning: Prediction file {pred_file} does not exist. Skipping.")
            continue

        # 加载真实标签
        gt_boxes, gt_labels, _ = load_labels(gt_file, model_name)
        targets = [{"boxes": torch.tensor(gt_boxes), "labels": torch.tensor(gt_labels)}]

        # 加载预测结果（带有 scores）
        pred_boxes, pred_labels, pred_scores = load_labels(pred_file, model_name, is_prediction=True)
        preds = [{"boxes": torch.tensor(pred_boxes), "labels": torch.tensor(pred_labels), "scores": torch.tensor(pred_scores)}]

        # 更新 mAP 计算器
        map_calculator.update(preds=preds, target=targets)

    # 计算并打印 mAP
    metrics = map_calculator.compute()
    metrics_results[model_name] = metrics
    print(f"{model_name} 模型的指标：mAP = {metrics['map']}")

    # 重置 mAP 计算器
    map_calculator.reset()

# 输出最终结果
print("各模型的评价指标:")
for model_name, metrics in metrics_results.items():
    print(f"{model_name}: mAP={metrics['map']:.4f}")
