from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import torch
import numpy as np
import cv2
import onnxruntime as ort
from opts import opts
from models.model import create_model, load_model
from detectors.detector_factory import detector_factory


def preprocess_image_pytorch_style(image_path, opt):
    """使用PyTorch检测器的预处理方式"""
    # 创建检测器
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    # 读取图像
    image = cv2.imread(image_path)

    # 使用检测器的预处理
    images, meta = detector.pre_process(image, scale=1.)

    return images, meta, image


def preprocess_image_onnx_style(image_path, input_h, input_w, mean, std):
    """使用我们ONNX转换脚本的预处理方式"""
    import cv2
    import numpy as np

    # 读取图像 (BGR格式)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    height, width = img.shape[:2]
    print(f"原始图像尺寸: {width}x{height}")

    # CenterNet预处理流程
    from convert_to_onnx import get_affine_transform

    c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    s = max(height, width) * 1.0

    print(f"中心点: {c}, 缩放: {s}")

    # 获取仿射变换矩阵
    trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
    print(f"仿射变换矩阵:\n{trans_input}")

    # 应用仿射变换
    inp_image = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    # 标准化 (保持BGR格式)
    inp_image = ((inp_image / 255.0 - mean) / std).astype(np.float32)

    # 转换为CHW格式并添加batch维度
    inp_image = inp_image.transpose(2, 0, 1)
    inp_image = np.expand_dims(inp_image, axis=0)

    return torch.from_numpy(inp_image), img


def compare_preprocessing(image_path, opt):
    """对比两种预处理方式的结果"""
    print("="*50)
    print("对比预处理结果")
    print("="*50)

    # PyTorch检测器预处理
    print("\n1. PyTorch检测器预处理:")
    pytorch_images, meta, original_img = preprocess_image_pytorch_style(image_path, opt)
    print(f"PyTorch输出shape: {pytorch_images.shape}")
    print(f"PyTorch输出dtype: {pytorch_images.dtype}")
    print(f"PyTorch输出范围: [{pytorch_images.min():.4f}, {pytorch_images.max():.4f}]")
    print(f"Meta信息: {meta}")

    # ONNX风格预处理
    print("\n2. ONNX风格预处理:")
    mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    onnx_images, _ = preprocess_image_onnx_style(image_path, opt.input_h, opt.input_w, mean, std)
    print(f"ONNX输出shape: {onnx_images.shape}")
    print(f"ONNX输出dtype: {onnx_images.dtype}")
    print(f"ONNX输出范围: [{onnx_images.min():.4f}, {onnx_images.max():.4f}]")

    # 计算差异
    print("\n3. 预处理差异分析:")
    if pytorch_images.shape == onnx_images.shape:
        diff = torch.abs(pytorch_images - onnx_images)
        print(f"最大差异: {diff.max():.6f}")
        print(f"平均差异: {diff.mean():.6f}")
        print(f"差异标准差: {diff.std():.6f}")

        # 检查是否相同
        if torch.allclose(pytorch_images, onnx_images, atol=1e-6):
            print("✅ 预处理结果一致!")
        else:
            print("❌ 预处理结果不一致!")

            # 分析差异原因
            print("\n差异分析:")
            for ch in range(3):
                ch_diff = diff[0, ch].max()
                print(f"通道{ch}最大差异: {ch_diff:.6f}")
    else:
        print(f"❌ 输出shape不匹配: PyTorch {pytorch_images.shape} vs ONNX {onnx_images.shape}")

    return pytorch_images, onnx_images, original_img


def compare_model_outputs(pytorch_images, onnx_images, opt):
    """对比模型输出"""
    print("\n" + "="*50)
    print("对比模型输出结果")
    print("="*50)

    # 1. PyTorch模型推理
    print("\n1. PyTorch模型推理:")
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model.eval()

    with torch.no_grad():
        pytorch_outputs = model(pytorch_images)

    print(f"PyTorch输出类型: {type(pytorch_outputs)}")
    if isinstance(pytorch_outputs, list) and len(pytorch_outputs) > 0:
        pytorch_out = pytorch_outputs[0]
        print(f"PyTorch输出keys: {pytorch_out.keys()}")
        for key, value in pytorch_out.items():
            print(f"  {key}: {value.shape}, 范围[{value.min():.4f}, {value.max():.4f}]")

    # 2. ONNX模型推理
    print("\n2. ONNX模型推理:")
    onnx_path = f"debug_model_{opt.arch}.onnx"

    # 如果ONNX模型不存在,先转换
    if not os.path.exists(onnx_path):
        print("转换ONNX模型...")
        dummy_input = torch.randn(1, 3, opt.input_h, opt.input_w)
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=11,
            input_names=['input'],
            output_names=list(opt.heads.keys())
        )

    # ONNX推理
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: onnx_images.numpy()}
    onnx_outputs = ort_session.run(None, ort_inputs)

    print(f"ONNX输出数量: {len(onnx_outputs)}")
    for i, (key, output) in enumerate(zip(opt.heads.keys(), onnx_outputs)):
        print(f"  {key}: {output.shape}, 范围[{output.min():.4f}, {output.max():.4f}]")

    # 3. 对比输出差异
    print("\n3. 模型输出差异分析:")
    if isinstance(pytorch_outputs, list) and len(pytorch_outputs) > 0:
        pytorch_out = pytorch_outputs[0]

        for i, key in enumerate(opt.heads.keys()):
            if key in pytorch_out:
                pytorch_tensor = pytorch_out[key].numpy()
                onnx_tensor = onnx_outputs[i]

                diff = np.abs(pytorch_tensor - onnx_tensor)
                print(f"\n{key}差异:")
                print(f"  最大差异: {diff.max():.8f}")
                print(f"  平均差异: {diff.mean():.8f}")
                print(f"  相对差异: {(diff / (np.abs(pytorch_tensor) + 1e-8)).max():.8f}")

                if np.allclose(pytorch_tensor, onnx_tensor, atol=1e-5):
                    print(f"  ✅ {key} 输出一致!")
                else:
                    print(f"  ❌ {key} 输出存在差异!")

    return pytorch_outputs, onnx_outputs


def analyze_detection_outputs(pytorch_outputs, onnx_outputs, opt, original_img):
    """分析检测输出并对比后处理结果"""
    print("\n" + "="*50)
    print("分析检测输出")
    print("="*50)

    if isinstance(pytorch_outputs, list) and len(pytorch_outputs) > 0:
        pytorch_out = pytorch_outputs[0]

        # 分析heatmap
        hm_pytorch = pytorch_out['hm'].numpy()
        hm_onnx = onnx_outputs[0]  # 假设第一个输出是hm

        print(f"\n1. Heatmap分析:")
        print(f"PyTorch heatmap shape: {hm_pytorch.shape}")
        print(f"ONNX heatmap shape: {hm_onnx.shape}")

        # 应用sigmoid
        hm_pytorch_sig = 1 / (1 + np.exp(-hm_pytorch))
        hm_onnx_sig = 1 / (1 + np.exp(-hm_onnx))

        print(f"PyTorch sigmoid范围: [{hm_pytorch_sig.min():.4f}, {hm_pytorch_sig.max():.4f}]")
        print(f"ONNX sigmoid范围: [{hm_onnx_sig.min():.4f}, {hm_onnx_sig.max():.4f}]")

        # 找出最高置信度
        pytorch_max_idx = np.unravel_index(hm_pytorch_sig.argmax(), hm_pytorch_sig.shape)
        onnx_max_idx = np.unravel_index(hm_onnx_sig.argmax(), hm_onnx_sig.shape)

        print(f"PyTorch最高置信度: {hm_pytorch_sig[pytorch_max_idx]:.4f} at {pytorch_max_idx}")
        print(f"ONNX最高置信度: {hm_onnx_sig[onnx_max_idx]:.4f} at {onnx_max_idx}")

        # 统计高置信度点数量
        threshold = 0.3
        pytorch_peaks = (hm_pytorch_sig > threshold).sum()
        onnx_peaks = (hm_onnx_sig > threshold).sum()

        print(f"高于{threshold}的点数量: PyTorch {pytorch_peaks}, ONNX {onnx_peaks}")

        # 分析每个类别的最大值
        print(f"\n2. 各类别最大置信度对比:")
        for cls in range(min(10, hm_pytorch.shape[1])):  # 只显示前10个类别
            pytorch_cls_max = hm_pytorch_sig[0, cls].max()
            onnx_cls_max = hm_onnx_sig[0, cls].max()
            print(f"类别 {cls}: PyTorch {pytorch_cls_max:.4f}, ONNX {onnx_cls_max:.4f}")


def main():
    # 解析参数
    opt = opts().parse()

    # 设置默认值
    if not hasattr(opt, 'heads') or opt.heads is None:
        from datasets.dataset_factory import get_dataset
        Dataset = get_dataset(opt.dataset, opt.task)
        opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    # 测试图像
    test_image = "readme/det1.png"
    if not os.path.exists(test_image):
        print(f"测试图像不存在: {test_image}")
        return

    print(f"测试图像: {test_image}")
    print(f"模型架构: {opt.arch}")
    print(f"输入尺寸: {opt.input_h}x{opt.input_w}")
    print(f"输出头: {opt.heads}")
    print(f"Mean: {opt.mean}")
    print(f"Std: {opt.std}")

    # 1. 对比预处理
    pytorch_images, onnx_images, original_img = compare_preprocessing(test_image, opt)

    # 2. 对比模型输出
    pytorch_outputs, onnx_outputs = compare_model_outputs(pytorch_images, onnx_images, opt)

    # 3. 分析检测输出
    analyze_detection_outputs(pytorch_outputs, onnx_outputs, opt, original_img)


if __name__ == '__main__':
    main()