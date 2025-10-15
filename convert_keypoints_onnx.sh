
source ~/anaconda3/bin/activate && \
conda activate py310 && \

# python src/convert_to_onnx.py multi_pose \
#  --dataset yolo_dataset \
#  --load_model exp/multi_pose/my_pose_experiment/model_best.pth \
#  --arch res_18 \
#  --yolo_num_kpts 4 \
#  --yolo_force_num_classes 2 \
#  --input_h 320 --input_w 640 \
#  --vis_thresh 0.1 \
#  --save_vis \
#  --save_heatmaps \
#  --yolo_mean "0 0 0" \
#  --yolo_std "1 1 1" \
#  --input_rgb \
#  --test_image /home/qwer/test.jpg
# #  --test_image data/yolo_dataset/images/train/1719370579616258600.jpg \
# #  --test_label data/yolo_dataset/labels/train/1719370579616258600.txt \

python src/convert_to_onnx.py multi_pose \
 --dataset yolo_dataset \
 --load_model exp/multi_pose/my_pose_experiment/model_last.pth \
 --arch res_18 \
 --yolo_num_kpts 10 \
 --input_h 320 --input_w 640 \
 --test_image /home/qwer/charger_program/CenterNet/data/yolo_annotations/images/train/140.jpg \
 --vis_thresh 0.1 \
 --save_vis \
 --save_heatmaps \
 --yolo_mean "0 0 0" \
 --yolo_std "1 1 1" \
 --input_rgb \
 --cross_class_nms_thresh 0.5
#  --test_label data/yolo_dataset/labels/train/1719370579616258600.txt \
