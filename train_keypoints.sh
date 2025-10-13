python src/main.py multi_pose --dataset yolo_dataset \
--exp_id my_pose_experiment --num_epochs 140 \
--val_debug_batches 10 --val_intervals 10 \
  --yolo_num_kpts 4 \
  --input_h 320 --input_w 640 \
  --hm_hp_weight 2 \
  --batch_size 4 \
  --num_workers 8 \
  --gpus 0 \
  --flip 0. \
  --amp \
  --yolo_mean "0 0 0" \
  --yolo_std "1 1 1" \
  --input_rgb
  # --debug_aug_vis --num_workers 0
  # --yolo_flip_pairs 0-1,2-3
  