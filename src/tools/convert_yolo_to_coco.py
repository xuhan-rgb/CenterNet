#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import cv2
import glob
import argparse
from datetime import datetime

def convert_yolo_to_coco(yolo_data_dir, output_dir, split='train'):
    """
    Convert YOLO format dataset to COCO format

    Args:
        yolo_data_dir: Directory containing YOLO format data
        output_dir: Directory to save COCO format annotations
        split: Dataset split ('train', 'val', 'test')
    """

    # YOLO directory structure:
    # yolo_data_dir/
    # ├── images/
    # │   ├── train/
    # │   ├── val/
    # │   └── test/
    # ├── labels/
    # │   ├── train/
    # │   ├── val/
    # │   └── test/
    # └── classes.txt

    img_dir = os.path.join(yolo_data_dir, 'images', split)
    label_dir = os.path.join(yolo_data_dir, 'labels', split)
    classes_file = os.path.join(yolo_data_dir, 'classes.txt')

    if not os.path.exists(img_dir):
        print(f"Images directory not found: {img_dir}")
        return

    if not os.path.exists(label_dir):
        print(f"Labels directory not found: {label_dir}")
        return

    # Load class names
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        print(f"Classes file not found: {classes_file}")
        print("Using default COCO classes")
        class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                      'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                      'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                      'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                      'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                      'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                      'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                      'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                      'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                      'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                      'toothbrush']

    # Initialize COCO format structure
    coco_format = {
        "info": {
            "description": f"YOLO dataset converted to COCO format - {split}",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "YOLO to COCO converter",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown License",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Add categories
    for i, class_name in enumerate(class_names):
        coco_format["categories"].append({
            "id": i + 1,
            "name": class_name,
            "supercategory": "object"
        })

    # Get all image files
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for ext in img_extensions:
        img_files.extend(glob.glob(os.path.join(img_dir, ext)))
        img_files.extend(glob.glob(os.path.join(img_dir, ext.upper())))

    img_files = sorted(img_files)
    annotation_id = 1

    print(f"Converting {len(img_files)} images from {split} split...")

    for img_id, img_path in enumerate(img_files):
        # Load image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue

        height, width = img.shape[:2]
        filename = os.path.basename(img_path)

        # Add image info
        coco_format["images"].append({
            "id": img_id + 1,
            "width": width,
            "height": height,
            "file_name": filename,
            "license": 1
        })

        # Load corresponding label file
        img_name = os.path.splitext(filename)[0]
        label_path = os.path.join(label_dir, img_name + '.txt')

        if not os.path.exists(label_path):
            print(f"Warning: No label file found for {filename}")
            continue

        # Read YOLO annotations
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                print(f"Warning: Invalid annotation format in {label_path}: {line}")
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                print(f"Warning: Invalid annotation values in {label_path}: {line}")
                continue

            # Convert YOLO format to COCO format
            # YOLO: normalized center coordinates (x_center, y_center, width, height)
            # COCO: absolute top-left coordinates (x, y, width, height)

            x_center_abs = x_center * width
            y_center_abs = y_center * height
            w_abs = w * width
            h_abs = h * height

            x = x_center_abs - w_abs / 2
            y = y_center_abs - h_abs / 2

            # Ensure coordinates are within image bounds
            x = max(0, min(x, width))
            y = max(0, min(y, height))
            w_abs = min(w_abs, width - x)
            h_abs = min(h_abs, height - y)

            if w_abs > 0 and h_abs > 0 and class_id < len(class_names):
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id + 1,
                    "category_id": class_id + 1,  # COCO categories are 1-indexed
                    "bbox": [x, y, w_abs, h_abs],
                    "area": w_abs * h_abs,
                    "iscrowd": 0
                })
                annotation_id += 1

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save COCO format annotations
    output_file = os.path.join(output_dir, f'instances_{split}2017.json')
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=2)

    print(f"Conversion completed!")
    print(f"Images processed: {len(coco_format['images'])}")
    print(f"Annotations created: {len(coco_format['annotations'])}")
    print(f"Categories: {len(coco_format['categories'])}")
    print(f"Output saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO format dataset to COCO format')
    parser.add_argument('yolo_data_dir', help='Path to YOLO dataset directory')
    parser.add_argument('--output_dir', default='./coco_annotations',
                       help='Output directory for COCO annotations (default: ./coco_annotations)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                       help='Dataset splits to convert (default: train val)')

    args = parser.parse_args()

    if not os.path.exists(args.yolo_data_dir):
        print(f"Error: YOLO data directory not found: {args.yolo_data_dir}")
        return

    for split in args.splits:
        print(f"\n=== Converting {split} split ===")
        convert_yolo_to_coco(args.yolo_data_dir, args.output_dir, split)

if __name__ == '__main__':
    main()