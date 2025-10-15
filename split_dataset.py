#!/usr/bin/env python
import os
import shutil
import random
import argparse

# 命令行参数配置
parser = argparse.ArgumentParser(description="Split dataset into train and validation sets")
parser.add_argument(
    "--dataset_dir", type=str, default="data/yolo_dataset", help="Dataset directory path (default: data/yolo_dataset)"
)
parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio (default: 0.8)")
parser.add_argument(
    "--source_split",
    type=str,
    default="images/trainval",
    help="Source split folder. Examples: 'images/trainval' (paired with labels/trainval), "
    "'trainval' (under images/labels), or 'images' (paired with label/).",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
args = parser.parse_args()

# 配置路径
dataset_dir = args.dataset_dir
# 支持两种源目录结构：
# 1. image/label (旧格式，用于yolo_dataset)
# 2. images/trainval, labels/trainval (新格式，用于yolo_annotations)
if args.source_split == "trainval":
    image_dir = os.path.join(dataset_dir, "images", "trainval")
    label_dir = os.path.join(dataset_dir, "labels", "trainval")
elif args.source_split.startswith("images/"):
    split_name = args.source_split.split("/", 1)[1]
    image_dir = os.path.join(dataset_dir, args.source_split)
    label_dir = os.path.join(dataset_dir, "labels", split_name)
elif args.source_split in ("train", "val", "test"):
    image_dir = os.path.join(dataset_dir, "images", args.source_split)
    label_dir = os.path.join(dataset_dir, "labels", args.source_split)
else:
    image_dir = os.path.join(dataset_dir, args.source_split)
    label_dir = os.path.join(dataset_dir, "label")

train_image_dir = os.path.join(dataset_dir, "images", "train")
train_label_dir = os.path.join(dataset_dir, "labels", "train")
val_image_dir = os.path.join(dataset_dir, "images", "val")
val_label_dir = os.path.join(dataset_dir, "labels", "val")

# 清空之前生成的目录
for dir_path in [train_image_dir, train_label_dir, val_image_dir, val_label_dir]:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

# 获取所有图像文件
if not os.path.exists(image_dir):
    print(f"Error: Image directory not found: {image_dir}")
    exit(1)

image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
print(f"Found {len(image_files)} images in {image_dir}")

# 随机打乱（设置种子以确保可重复性）
random.seed(args.seed)
random.shuffle(image_files)

# 划分训练集和验证集
train_split = int(args.train_ratio * len(image_files))
train_files = image_files[:train_split]
val_files = image_files[train_split:]

print(f"Dataset directory: {dataset_dir}")
print(f"Training ratio: {args.train_ratio}")
print(f"Training set: {len(train_files)} images")
print(f"Validation set: {len(val_files)} images")

# 复制训练集
for img_file in train_files:
    base_name = os.path.splitext(img_file)[0]

    # 复制图像
    src_img = os.path.join(image_dir, img_file)
    dst_img = os.path.join(train_image_dir, img_file)
    shutil.copy2(src_img, dst_img)

    # 复制标签
    label_file = base_name + ".txt"
    src_label = os.path.join(label_dir, label_file)
    dst_label = os.path.join(train_label_dir, label_file)
    if os.path.exists(src_label):
        shutil.copy2(src_label, dst_label)

# 复制验证集
for img_file in val_files:
    base_name = os.path.splitext(img_file)[0]

    # 复制图像
    src_img = os.path.join(image_dir, img_file)
    dst_img = os.path.join(val_image_dir, img_file)
    shutil.copy2(src_img, dst_img)

    # 复制标签
    label_file = base_name + ".txt"
    src_label = os.path.join(label_dir, label_file)
    dst_label = os.path.join(val_label_dir, label_file)
    if os.path.exists(src_label):
        shutil.copy2(src_label, dst_label)

print(f"Dataset split completed!")
print(f"Training images saved to: {train_image_dir}")
print(f"Training labels saved to: {train_label_dir}")
print(f"Validation images saved to: {val_image_dir}")
print(f"Validation labels saved to: {val_label_dir}")
