import os
import xml.etree.ElementTree as ET
import random
import shutil

# XML文件目录和图片文件目录
xml_dir = '/content/Exp3-Yolo/Annotations'  # 替换为你XML文件的目录
img_dir = '/content/Exp3-Yolo/images'  # 替换为图片文件的目录
# 输出目录
output_base_dir = '/content/Exp3-Yolo'
train_label_dir = os.path.join(output_base_dir, 'train/labels')
val_label_dir = os.path.join(output_base_dir, 'val/labels')
test_label_dir = os.path.join(output_base_dir, 'test/labels')
train_image_dir = os.path.join(output_base_dir, 'train/images')
val_image_dir = os.path.join(output_base_dir, 'val/images')
test_image_dir = os.path.join(output_base_dir, 'test/images')

# 确保目录存在
for dir_path in [train_label_dir, val_label_dir, test_label_dir, train_image_dir, val_image_dir, test_image_dir]:
    os.makedirs(dir_path, exist_ok=True)

# 列出所有XML文件
xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

# 设置随机种子
random.seed(42)  # 这里设定一个固定的种子，使得划分结果可以复现

# 划分比例
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# 随机打乱并划分
random.shuffle(xml_files)
train_files = xml_files[:int(train_ratio * len(xml_files))]
val_files = xml_files[int(train_ratio * len(xml_files)):int((train_ratio + val_ratio) * len(xml_files))]
test_files = xml_files[int((train_ratio + val_ratio) * len(xml_files)):]

# 类别字典（根据你的数据集自行调整）
class_mapping = {'equation': 0}  # 替换类别

# 转换函数
def convert_xml_to_yolo(xml_file, output_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)

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

            # 转换为YOLO格式 (x_center, y_center, width, height)
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # 写入YOLO格式
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# 处理并保存文件（包括图片文件）
def process_files(file_list, label_dir, image_dir):
    for xml_file in file_list:
        xml_path = os.path.join(xml_dir, xml_file)
        txt_file = xml_file.replace('.xml', '.txt')
        output_txt_path = os.path.join(label_dir, txt_file)

        # 转换并保存txt文件
        convert_xml_to_yolo(xml_path, output_txt_path)

        # 复制对应的图片文件到目标目录
        img_file = xml_file.replace('.xml', '.jpg')
        img_path = os.path.join(img_dir, img_file)
        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(image_dir, img_file))

# 运行转换和保存过程
process_files(train_files, train_label_dir, train_image_dir)
process_files(val_files, val_label_dir, val_image_dir)
process_files(test_files, test_label_dir, test_image_dir)

print("XML文件已成功转换为YOLO格式，并按7:1:2比例分配至train、val和test文件夹的labels和images文件夹中。")
