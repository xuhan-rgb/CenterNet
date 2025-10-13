
python src/convert_to_onnx.py ctdet \
    --dataset yolo_dataset \
    --load_model exp/ctdet/my_experiment/model_best.pth \
    --arch res_18 \
    --test_image data/yolo_dataset/images/val/1758799813548058000.jpg \
    --save_vis \
    --save_heatmaps