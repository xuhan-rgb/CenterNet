# ONNX 推理脚本使用说明

1. `source ~/anaconda3/bin/activate && conda activate py310`
2. `python infer_onnx_runtime.py --config onnx_infer.yaml /绝对路径/到/图片.jpg`

如需自定义模型或类别文件，编辑 `onnx_infer.yaml` 中的 `onnx_model` 与 `class_names` 路径。
