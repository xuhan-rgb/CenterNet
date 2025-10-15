#!/usr/bin/env python
"""
测试smart_crop是否真的避免了keypoint截断
"""
import sys
sys.path.insert(0, 'src/lib')

from opts import opts
from datasets.dataset_factory import get_dataset
import numpy as np

def test_crop_strategy(use_smart_crop=False):
    """测试crop策略，统计截断的keypoint数量"""

    # 设置选项
    args = [
        'multi_pose',
        '--dataset', 'yolo_dataset',
        '--input_h', '320',
        '--input_w', '640',
        '--yolo_num_kpts', '10',
        '--yolo_flip_pairs', '0-1,2-3,4-5,6-7',
        '--input_rgb',
        '--yolo_mean', '0 0 0',
        '--yolo_std', '1 1 1',
        '--learn_invisible_kpts',
    ]

    if use_smart_crop:
        args.append('--smart_crop')

    opt = opts().parse(args)

    # 创建数据集
    Dataset = get_dataset(opt.dataset, opt.task)

    # 需要先创建一个临时dataset来获取dataset spec
    temp_dataset = Dataset(opt, 'train')
    opt = opts().update_dataset_info_and_set_heads(opt, temp_dataset)

    # 重新创建dataset (使用更新后的opt)
    dataset = Dataset(opt, 'train')

    # 统计信息
    total_samples = min(100, len(dataset))  # 测试100个样本
    total_keypoints = 0
    truncated_keypoints = 0
    samples_with_truncation = 0

    print(f"\n{'='*60}")
    print(f"测试 {'Smart Crop' if use_smart_crop else 'Random Crop'} 策略")
    print(f"{'='*60}\n")

    for i in range(total_samples):
        sample = dataset[i]

        # 检查hps (keypoint positions)
        hps = sample['hps']  # shape: (max_objs, num_joints*2)
        hps_mask = sample['hps_mask']  # shape: (max_objs, num_joints*2)

        output_w = opt.output_w
        output_h = opt.output_h

        sample_has_truncation = False

        for obj_idx in range(hps.shape[0]):
            for joint_idx in range(opt.num_joints):
                x_idx = joint_idx * 2
                y_idx = joint_idx * 2 + 1

                # 只检查有mask的keypoint
                if hps_mask[obj_idx, x_idx] > 0:
                    total_keypoints += 1

                    # 获取实际坐标 (相对于center，需要加上center位置)
                    ind = sample['ind'][obj_idx]
                    ct_y = ind // output_w
                    ct_x = ind % output_w

                    kp_x = ct_x + hps[obj_idx, x_idx]
                    kp_y = ct_y + hps[obj_idx, y_idx]

                    # 检查是否超出边界
                    if kp_x < 0 or kp_x >= output_w or kp_y < 0 or kp_y >= output_h:
                        truncated_keypoints += 1
                        sample_has_truncation = True
                        print(f"  样本 {i}: keypoint {joint_idx} 截断 (x={kp_x:.2f}, y={kp_y:.2f}, bounds=[0-{output_w}, 0-{output_h}])")

        if sample_has_truncation:
            samples_with_truncation += 1

    # 输出统计结果
    print(f"\n{'='*60}")
    print(f"统计结果:")
    print(f"  总样本数: {total_samples}")
    print(f"  总keypoint数: {total_keypoints}")
    print(f"  截断的keypoint数: {truncated_keypoints}")
    print(f"  截断率: {100.0 * truncated_keypoints / max(1, total_keypoints):.2f}%")
    print(f"  有截断的样本数: {samples_with_truncation}")
    print(f"  样本截断率: {100.0 * samples_with_truncation / total_samples:.2f}%")
    print(f"{'='*60}\n")

    return {
        'total_samples': total_samples,
        'total_keypoints': total_keypoints,
        'truncated_keypoints': truncated_keypoints,
        'samples_with_truncation': samples_with_truncation,
    }

if __name__ == '__main__':
    print("\n" + "="*60)
    print("测试数据增强是否导致keypoint截断")
    print("="*60)

    # 测试Random Crop
    print("\n[测试1] 使用Random Crop (原始策略)")
    results_random = test_crop_strategy(use_smart_crop=False)

    # 测试Smart Crop
    print("\n[测试2] 使用Smart Crop (新策略)")
    results_smart = test_crop_strategy(use_smart_crop=True)

    # 比较结果
    print("\n" + "="*60)
    print("对比结果:")
    print("="*60)
    print(f"Random Crop: {results_random['truncated_keypoints']}/{results_random['total_keypoints']} "
          f"({100.0*results_random['truncated_keypoints']/max(1,results_random['total_keypoints']):.2f}%) keypoints 截断")
    print(f"Smart Crop:  {results_smart['truncated_keypoints']}/{results_smart['total_keypoints']} "
          f"({100.0*results_smart['truncated_keypoints']/max(1,results_smart['total_keypoints']):.2f}%) keypoints 截断")

    if results_smart['truncated_keypoints'] == 0:
        print("\n✅ Smart Crop 成功避免了所有keypoint截断!")
    elif results_smart['truncated_keypoints'] < results_random['truncated_keypoints']:
        reduction = 100.0 * (1 - results_smart['truncated_keypoints'] / max(1, results_random['truncated_keypoints']))
        print(f"\n✅ Smart Crop 减少了 {reduction:.1f}% 的截断!")
    else:
        print("\n⚠️ Smart Crop 没有改善截断情况")
    print("="*60 + "\n")
