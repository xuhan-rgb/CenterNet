
source ~/anaconda3/bin/activate && \
conda activate py310 && \

python src/convert_to_onnx.py multi_pose \
 --dataset yolo_dataset \
 --load_model exp/multi_pose/my_pose_experiment/model_best.pth \
 --arch res_18 \
 --yolo_num_kpts 4 \
 --input_h 320 --input_w 640 \
 --test_image /home/qwer/test.jpg \
 --vis_thresh 0.1 \
 --save_vis \
 --save_heatmaps \
 --yolo_mean "0 0 0" \
 --yolo_std "1 1 1" \
 --input_rgb \
#  --test_label data/yolo_dataset/labels/train/1719370579616258600.txt \
