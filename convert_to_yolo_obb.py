import os
import cv2
import xml.etree.ElementTree as ET
import random
import shutil
import numpy as np

# 设置随机种子
random.seed(42)

# 设置原始XML和图像目录路径
xml_dir = './Annotations'  # XML文件的目录
img_dir = './images'  # 图片文件的目录
output_base_dir = './obb_dataset'  # 生成的obb_dataset目录

# 输出目录结构
train_label_dir = os.path.join(output_base_dir, 'train/labels')
val_label_dir = os.path.join(output_base_dir, 'val/labels')
test_label_dir = os.path.join(output_base_dir, 'test/labels')
train_image_dir = os.path.join(output_base_dir, 'train/images')
val_image_dir = os.path.join(output_base_dir, 'val/images')
test_image_dir = os.path.join(output_base_dir, 'test/images')

# 确保所有输出目录存在
for dir_path in [train_label_dir, val_label_dir, test_label_dir, train_image_dir, val_image_dir, test_image_dir]:
    os.makedirs(dir_path, exist_ok=True)

# 列出所有XML文件并划分数据集
xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
random.shuffle(xml_files)  # 使用随机种子42进行打乱
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2
train_files = xml_files[:int(train_ratio * len(xml_files))]
val_files = xml_files[int(train_ratio * len(xml_files)):int((train_ratio + val_ratio) * len(xml_files))]
test_files = xml_files[int((train_ratio + val_ratio) * len(xml_files)):]

# 类别映射字典
class_mapping = {'equation': 0}  # 根据实际类别修改

# 查找图片文件路径，兼容 .jpg 和 .JPG
def find_image_path(base_name, img_dir):
    for ext in ['.jpg', '.JPG']:
        img_path = os.path.join(img_dir, base_name + ext)
        if os.path.exists(img_path):
            return img_path
    return None

# 转换 XML 为 YOLO OBB 格式的函数
def convert_xml_to_yolo_obb(xml_file, output_file, img_width, img_height, rotation_matrix=None):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    with open(output_file, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_mapping:
                continue
            
            class_id = class_mapping[class_name]
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # 计算四个角点坐标
            points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32)

            # 如果有旋转矩阵，应用旋转变换
            if rotation_matrix is not None:
                points = cv2.transform(np.array([points]), rotation_matrix)[0]

            # 将坐标归一化到0-1范围
            normalized_points = [(x / img_width, y / img_height) for x, y in points]
            normalized_points_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in normalized_points])

            # 写入YOLO OBB格式
            f.write(f"{class_id} {normalized_points_str}\n")

# 应用旋转变换并保存图像和标签
def process_files_obb(file_list, label_dir, image_dir, angle=15):
    for xml_file in file_list:
        xml_path = os.path.join(xml_dir, xml_file)
        base_name = xml_file.replace('.xml', '')
        
        # 查找图片路径，支持 .jpg 和 .JPG 扩展名
        img_path = find_image_path(base_name, img_dir)
        if img_path is None:
            print(f"Warning: 找不到对应的图像文件：{base_name}.jpg 或 {base_name}.JPG")
            continue
        
        # 读取图像并获取尺寸
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: 无法读取图像文件 {img_path}")
            continue
        h, w = image.shape[:2]

        # 创建旋转矩阵
        center = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 对图像应用旋转
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

        # 保存旋转后的图像
        output_image_path = os.path.join(image_dir, base_name + '.jpg')
        cv2.imwrite(output_image_path, rotated_image)

        # 转换并保存YOLO OBB格式标签
        output_txt_path = os.path.join(label_dir, base_name + '.txt')
        convert_xml_to_yolo_obb(xml_path, output_txt_path, w, h, rotation_matrix)

# 运行转换和保存过程
process_files_obb(train_files, train_label_dir, train_image_dir)
process_files_obb(val_files, val_label_dir, val_image_dir)
process_files_obb(test_files, test_label_dir, test_image_dir)

print("XML文件已成功转换为YOLO OBB格式，并按7:1:2比例分配至obb_dataset下的train、val和test文件夹的labels和images文件夹中。")
