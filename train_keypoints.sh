
rm -r exp/multi_pose/my_pose_experiment/*
# python src/main.py multi_pose --dataset yolo_dataset \
# --exp_id my_pose_experiment --num_epochs 140 \
# --val_debug_batches 10 --val_intervals 10 \
#   --yolo_num_kpts 4 \
#   --yolo_force_num_classes 2 \
#   --yolo_dataset_dir data/yolo_dataset \
#   --input_h 320 --input_w 640 \
#   --hm_hp_weight 2 \
#   --batch_size 4 \
#   --num_workers 8 \
#   --gpus 0 \
#   --flip 0. \
#   --amp \
#   --yolo_mean "0 0 0" \
#   --yolo_std "1 1 1" \
#   --input_rgb \
#   --learn_invisible_kpts \
#   --learn_truncated_kpts
#   # --debug_aug_vis --num_workers 0
#   # --yolo_flip_pairs 0-1,2-3

python src/main.py multi_pose --dataset yolo_dataset \
  --exp_id my_pose_experiment --num_epochs 140 \
  --val_debug_batches 10 --val_intervals 10 \
  --yolo_num_kpts 10 \
  --yolo_force_num_classes 22 \
  --yolo_dataset_dir data/yolo_annotations/ \
  --input_h 320 --input_w 640 \
  --hm_hp_weight 2 \
  --batch_size 16 \
  --num_workers 8 \
  --gpus 0 \
  --flip 0. \
  --amp \
  --yolo_mean "0 0 0" \
  --yolo_std "1 1 1" \
  --input_rgb \
  --learn_invisible_kpts \
  --learn_truncated_kpts \
  --keep_bbox_without_kpts \
  --no_color_aug \
  --yolo_repeat_factor 5
  # --not_rand_crop 
  # --log_sample_details
  # --debug_aug_vis --num_workers 0
  # --yolo_flip_pairs 0-1,2-3


# 关键点学习策略说明:
#   | 关键点类型       | visibility | 位置loss                   | 可见性loss | 说明 |
#   |----------------|------------|--------------------------|---------|------|
#   | 无效点          | < 0        | ❌                        | ❌       | 不存在的点，完全跳过 |
#   | 标注不可见       | = 0        | 根据--learn_invisible_kpts | ✅       | 遮挡但位置已知 |
#   | 标注可见+在边界内 | > 0        | ✅                        | ✅       | 正常可见点 |
#   | 截断点           | > 0→0      | 根据--learn_truncated_kpts | ✅       | 超出边界，标记为不可见 |

# 当前配置 (启用两个标志):
#   - --learn_invisible_kpts: 标注不可见的点(vis=0)也学习位置
#   - --learn_truncated_kpts: 截断的点也学习位置

# 效果: 最大化利用标注信息，提高模型对遮挡和边界情况的鲁棒性

# 传了 --not_reg_bbox → wh 头及 loss 整体都不存在，无论是否有关键点。
# 不传 --not_reg_bbox，但没开 --keep_bbox_without_kpts → 只有当存在至少一个可见关键点时该框才会参与宽高训练，缺少关键点的盒子被跳过。
# 不传 --not_reg_bbox，而且开了 --keep_bbox_without_kpts → 即使找不到可见关键点，该框仍会参与宽高训练；这是你当前需要的行为。