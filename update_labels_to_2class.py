#!/usr/bin/env python3
"""
更新YOLO标签文件从3分类改为2分类
原始类别：0=person, 1=bicycle, 2=car
新类别：0=person, 1=vehicle (bicycle + car)
"""

import os
import glob

def update_label_file(file_path):
    """更新单个标签文件"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    modified = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) >= 5:
            class_id = int(float(parts[0]))

            # 将bicycle(1)和car(2)都映射到vehicle(1)
            if class_id == 2:  # car -> vehicle
                parts[0] = '1'
                modified = True
            elif class_id == 1:  # bicycle -> vehicle (保持为1)
                pass  # 已经是1，不需要修改
            # class_id == 0 (person) 保持不变

            updated_lines.append(' '.join(parts) + '\n')

    if modified:
        with open(file_path, 'w') as f:
            f.writelines(updated_lines)
        print(f"Updated: {file_path}")

    return modified

def main():
    """主函数"""
    label_dirs = [
        'data/yolo_dataset/labels/train',
        'data/yolo_dataset/labels/val'
    ]

    total_updated = 0

    for label_dir in label_dirs:
        if os.path.exists(label_dir):
            label_files = glob.glob(os.path.join(label_dir, '*.txt'))
            print(f"Processing {len(label_files)} files in {label_dir}")

            for label_file in label_files:
                if update_label_file(label_file):
                    total_updated += 1
        else:
            print(f"Directory not found: {label_dir}")

    print(f"\n总计更新了 {total_updated} 个标签文件")

    # 验证更新结果
    print("\n验证更新结果:")
    for label_dir in label_dirs:
        if os.path.exists(label_dir):
            print(f"\n{label_dir} 中的类别分布:")
            os.system(f"find {label_dir} -name '*.txt' -exec grep -h '^[01]' {{}} \\; | cut -d' ' -f1 | sort | uniq -c")

if __name__ == "__main__":
    main()