from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import numpy as np
import torch
import torch.onnx
import argparse
import colorsys
from opts import opts
from models.model import create_model, load_model
from models.decode import ctdet_decode, multi_pose_decode
from utils.post_process import ctdet_post_process, multi_pose_post_process
import matplotlib.pyplot as plt

from dev_common import set_chinese_font, ColorLogger

logger = ColorLogger()

COLOR_MAP = {
    "reset": "\033[0m",
    "info": "\033[92m",
    "warn": "\033[93m",
    "error": "\033[91m",
    "header": "\033[95m",
    "detail": "\033[96m",
    "gt": "\033[94m",
    "pred": "\033[92m",
    "delta": "\033[91m",
}

FORCED_NUM_CLASSES = 2
FORCED_CLASS_NAMES = [f"class_{i}" for i in range(FORCED_NUM_CLASSES)]


def _colorize(msg, color_key=None):
    if color_key and color_key in COLOR_MAP:
        return f"{COLOR_MAP[color_key]}{msg}{COLOR_MAP['reset']}"
    return msg


def log_info(msg, color_key=None):
    print(_colorize(msg, color_key))


def log_warn(msg):
    print(_colorize(msg, "warn"))


def log_error(msg):
    print(_colorize(msg, "error"))


def _generate_color_palette(count):
    if count <= 0:
        return []
    palette = []
    for idx in range(count):
        hue = float(idx) / max(count, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


def _ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _format_number(value):
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    return f"{float(value):.8f}"


def _save_array_to_txt(file_path, array):
    array_np = np.asarray(array)
    with open(file_path, "w", encoding="ascii") as handle:
        flat = array_np.reshape(-1)
        for value in flat:
            handle.write(f"{_format_number(value)}\n")


def _save_detections_to_txt(file_path, detections):
    with open(file_path, "w", encoding="ascii") as handle:
        json.dump(detections, handle, ensure_ascii=True, indent=2)


def load_network_input_txt(file_path, expected_shape):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"network input file not found: {file_path}")
    if expected_shape is None:
        raise ValueError("expected_shape must be provided to load network input txt")
    expected_size = int(np.prod(expected_shape))

    with open(file_path, "r", encoding="ascii") as handle:
        values = []
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                values.append(float(stripped))
            except ValueError as exc:
                raise ValueError(f"invalid numeric value in {file_path}: {stripped}") from exc

    values_np = np.asarray(values, dtype=np.float32)
    if values_np.size != expected_size:
        raise ValueError(f"data size mismatch in {file_path}: expected {expected_size}, got {values_np.size}")
    return values_np.reshape(expected_shape)


def save_onnx_io_dump(base_name, input_tensor, output_names, ort_outputs, detections):
    result_dir = os.path.join(os.getcwd(), "result")
    _ensure_directory(result_dir)

    input_path = os.path.join(result_dir, f"{base_name}_network_input.txt")
    _save_array_to_txt(input_path, input_tensor)

    for idx, output_array in enumerate(ort_outputs):
        head_name = output_names[idx] if idx < len(output_names) else f"head_{idx}"
        output_path = os.path.join(result_dir, f"{base_name}_output_{head_name}.txt")
        _save_array_to_txt(output_path, output_array)

    detections_path = os.path.join(result_dir, f"{base_name}_detections.txt")
    _save_detections_to_txt(detections_path, detections)
    log_info(f"Saved ONNX IO dumps under: {result_dir}")


def convert_to_onnx(opt):
    """
    Convert CenterNet model to ONNX format
    """
    log_info("Creating model...")
    model = create_model(opt.arch, opt.heads, opt.head_conv)

    if opt.load_model != "":
        model = load_model(model, opt.load_model)
        log_info(f"Loaded model from: {opt.load_model}")

    model.eval()

    # Create dummy input tensor
    dummy_input = torch.randn(1, 3, opt.input_h, opt.input_w)

    # Set output path
    if opt.output_path == "":
        model_name = os.path.basename(opt.load_model).split(".")[0] if opt.load_model else "model"
        opt.output_path = f"{model_name}_{opt.arch}.onnx"

    log_info("Converting model to ONNX format...")
    log_info(f"Input shape: {tuple(dummy_input.shape)}")
    log_info(f"Output path: {opt.output_path}")

    # Convert to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        opt.output_path,
        export_params=True,
        opset_version=opt.opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=list(opt.heads.keys()),
        dynamic_axes=(
            {"input": {0: "batch_size"}, **{head: {0: "batch_size"} for head in opt.heads.keys()}}
            if opt.dynamic_batch
            else None
        ),
    )

    log_info(f"Model successfully converted to: {opt.output_path}")

    logger.info(f"ONNX model saved to: {opt.output_path}")
    logger.info(f"opt.verify={opt.verify}")
    opt.verify = True
    if opt.verify:
        log_info("Verifying ONNX model...")
        import onnx
        import onnxruntime as ort

        # Load and check the ONNX model
        onnx_model = onnx.load(opt.output_path)
        onnx.checker.check_model(onnx_model)
        log_info("ONNX model check passed!")

        # Compare outputs
        log_info("Comparing PyTorch and ONNX outputs...")
        ort_session = ort.InferenceSession(opt.output_path)

        # PyTorch inference
        with torch.no_grad():
            torch_out = model(dummy_input)

        # ONNX inference
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)

        # Compare outputs (PyTorch model returns [dict], we need the first element)
        torch_out_dict = torch_out[0]
        for i, head_name in enumerate(opt.heads.keys()):
            torch_output = torch_out_dict[head_name].numpy()
            onnx_output = ort_outs[i]
            max_diff = abs(torch_output - onnx_output).max()
            log_info(f"{head_name}: max difference = {max_diff:.6f}")

        log_info("Verification completed!")

    if opt.test_image and os.path.exists(opt.test_image):
        log_info(f"Testing ONNX model on image: {opt.test_image}")
        test_image_inference(model, opt)


def get_affine_transform(center, scale, rot, output_size, shift=None, inv=0):
    """获取仿射变换矩阵 - 从CenterNet utils复制"""
    import numpy as np
    import cv2

    if shift is None:
        shift = np.array([0, 0], dtype=np.float32)
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_dir(src_point, rot_rad):
    """获取方向向量"""
    import math

    sn, cs = math.sin(rot_rad), math.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def affine_transform(pt, t):
    """仿射变换单个点"""
    import numpy as np

    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    """获取第三个点"""
    import numpy as np

    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


import cv2


import cv2
import numpy as np


def preprocess_image(image_path, input_h, input_w, mean, std, use_rgb=False):
    """CenterNet标准预处理"""
    import cv2
    import numpy as np

    # 读取图像 (BGR格式)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    log_info(f"Input image shape / size: shape={img_bgr.shape}, input_h={input_h}, input_w={input_w}")

    original_img = img_bgr.copy()
    if use_rgb:
        print("================> use rgb")
        img_proc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_proc = img_bgr

    height, width = img_proc.shape[:2]

    # CenterNet预处理流程
    c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    s = max(height, width) * 1.0
    print("c = ", c, "s = ", s)

    # # 获取仿射变换矩阵
    trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
    print(trans_input)

    # 应用仿射变换（保持与训练/推理流程一致）
    # TODO: 需要优化这里
    inp_image = cv2.warpAffine(img_proc, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
    # inp_image = img_proc[60:1080, :, :]
    # inp_image = cv2.resize(inp_image, (input_w, input_h))

    # 标准化 (默认 BGR，如使用 RGB 输入需保持 mean/std 与之对应)
    print("=============> ", mean, std)
    inp_image = ((inp_image / 255.0 - mean) / std).astype(np.float32)

    # 转换为CHW格式并添加batch维度
    inp_image = inp_image.transpose(2, 0, 1)
    inp_image = np.expand_dims(inp_image, axis=0)

    return inp_image, original_img, (width, height), c, s


def postprocess_detections(outputs, opt, center, scale):
    """根据任务类型对ONNX输出进行后处理"""

    head_names = list(opt.heads.keys())
    head_map = {name: outputs[idx] for idx, name in enumerate(head_names)}

    def _to_tensor(name):
        if name not in head_map:
            return None
        tensor = torch.from_numpy(head_map[name]).float()
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor

    hm = _to_tensor("hm")
    if hm is None:
        raise ValueError("ONNX outputs missing 'hm' head")
    hm = torch.sigmoid(hm)

    score_thresh = getattr(opt, "vis_thresh", 0.3)
    results = []

    if opt.task == "ctdet":
        wh = _to_tensor("wh")
        if wh is None:
            raise ValueError("ONNX outputs missing 'wh' head for ctdet")
        reg = _to_tensor("reg") if "reg" in head_map else None
        cat_spec_wh = getattr(opt, "cat_spec_wh", False)

        if reg is not None:
            # regression head stays raw - identical to PyTorch pipeline
            pass

        dets = ctdet_decode(
            hm,
            wh,
            reg=reg,
            cat_spec_wh=cat_spec_wh,
            K=getattr(opt, "K", 100),
        )
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        processed = ctdet_post_process(
            dets.copy(),
            [center],
            [scale],
            hm.shape[3],
            hm.shape[2],
            opt.num_classes,
        )

        for cls_id, bboxes in processed[0].items():
            for bbox in bboxes:
                if len(bbox) < 5 or bbox[4] < score_thresh:
                    continue
                class_id = int(cls_id - 1)
                class_id = max(0, min(class_id, FORCED_NUM_CLASSES - 1))
                results.append(
                    {
                        "bbox": [float(b) for b in bbox[:4]],
                        "confidence": float(bbox[4]),
                        "class_id": class_id,
                    }
                )

    elif opt.task == "multi_pose":
        wh = _to_tensor("wh")
        hps = _to_tensor("hps")
        if wh is None or hps is None:
            raise ValueError("ONNX outputs missing multi-pose heads 'wh' or 'hps'")
        reg = _to_tensor("reg") if "reg" in head_map else None
        hm_hp = _to_tensor("hm_hp") if "hm_hp" in head_map else None
        hp_offset = _to_tensor("hp_offset") if "hp_offset" in head_map else None
        hp_vis = _to_tensor("hp_vis") if "hp_vis" in head_map else None

        if hm_hp is not None:
            hm_hp = torch.sigmoid(hm_hp)

        dets, initial_kps_raw = multi_pose_decode(
            hm,
            wh,
            hps,
            reg=reg if getattr(opt, "reg_offset", False) else None,
            hm_hp=hm_hp if getattr(opt, "hm_hp", False) else None,
            hp_offset=hp_offset if getattr(opt, "reg_hp_offset", False) else None,
            hp_vis=hp_vis,
            K=getattr(opt, "K", 100),
            log=True,
            return_initial_kps=True,
        )
        raw_dets = dets.detach().cpu().numpy()
        initial_kps_np = initial_kps_raw.detach().cpu().numpy()
        sample_count = min(raw_dets.shape[1], 5)
        # debug prints removed: raw dets shape and samples
        dets = raw_dets.reshape(1, -1, dets.shape[2])
        # debug prints removed: reg/hp_offset shapes
        processed = multi_pose_post_process(
            dets.copy(),
            [center],
            [scale],
            hm.shape[2],
            hm.shape[3],
            getattr(opt, "num_joints", None),
        )

        # 对initial_kps也进行坐标变换
        num_joints = getattr(opt, "num_joints", 0)
        initial_kps_transformed = []
        if num_joints > 0:
            from lib.utils.image import transform_preds

            for det_idx in range(initial_kps_np.shape[1]):
                kps_flat = initial_kps_np[0, det_idx, :]
                kps_reshaped = kps_flat.reshape(-1, 2)
                kps_trans = transform_preds(kps_reshaped, center, scale, (hm.shape[3], hm.shape[2]))
                initial_kps_list = [(float(kps_trans[j, 0]), float(kps_trans[j, 1])) for j in range(num_joints)]
                initial_kps_transformed.append(initial_kps_list)

        entry_counter = 0
        for cls_id, entries in processed[0].items():
            for entry_idx, entry in enumerate(entries):
                if len(entry) < 5 or entry[4] < score_thresh:
                    continue
                bbox = [float(b) for b in entry[:4]]
                keypoints = []
                visibilities = []
                if num_joints > 0:
                    kp_vals = entry[5 : 5 + num_joints * 2]
                    for j in range(num_joints):
                        keypoints.append(
                            (
                                float(kp_vals[2 * j]),
                                float(kp_vals[2 * j + 1]),
                            )
                        )
                    vis_start = 5 + num_joints * 2
                    vis_end = vis_start + num_joints
                    if len(entry) >= vis_end:
                        visibilities = [float(entry[vis_start + j]) for j in range(num_joints)]
                raw_idx = entry_idx
                if raw_idx < raw_dets.shape[1]:
                    raw_bbox_vals = raw_dets[0, raw_idx, :4]
                    raw_score_logit = raw_dets[0, raw_idx, 4]
                    raw_center_x = (raw_bbox_vals[0] + raw_bbox_vals[2]) / 2.0
                    raw_center_y = (raw_bbox_vals[1] + raw_bbox_vals[3]) / 2.0
                    raw_width = raw_bbox_vals[2] - raw_bbox_vals[0]
                    raw_height = raw_bbox_vals[3] - raw_bbox_vals[1]
                    score_sigmoid = 1.0 / (1.0 + np.exp(-raw_score_logit))
                    kept_rank = entry_counter + 1
                    log_info(
                        (
                            f"[decode->image] det {raw_idx} (kept #{kept_rank}): model_bbox="
                            f"[{raw_bbox_vals[0]:.2f}, {raw_bbox_vals[1]:.2f}, "
                            f"{raw_bbox_vals[2]:.2f}, {raw_bbox_vals[3]:.2f}]"
                            f", center=({raw_center_x:.2f}, {raw_center_y:.2f}), "
                            f"wh=({raw_width:.2f}, {raw_height:.2f}), score_sigmoid={score_sigmoid:.4f}"
                            f" -> image_bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"
                        ),
                        "detail",
                    )
                # debug prints removed: det summary, keypoints and offsets
                class_id = int(cls_id - 1)
                class_id = max(0, min(class_id, FORCED_NUM_CLASSES - 1))
                result_dict = {
                    "bbox": bbox,
                    "confidence": float(entry[4]),
                    "class_id": class_id,
                    "keypoints": keypoints,
                }
                if visibilities:
                    result_dict["keypoint_visibility"] = visibilities
                if raw_idx < len(initial_kps_transformed):
                    result_dict["initial_keypoints"] = initial_kps_transformed[raw_idx]
                results.append(result_dict)
                entry_counter += 1
    else:
        raise ValueError(f"Unsupported task '{opt.task}' for post-processing")

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results[:50]


def visualize_detections(image, detections, class_names=None, save_path=None, keypoint_edges=None):
    """可视化检测结果"""
    import cv2

    vis_img = image.copy()

    # 统一类别显示
    if class_names:
        class_names = list(class_names)[:FORCED_NUM_CLASSES]
        if len(class_names) < FORCED_NUM_CLASSES:
            class_names += [f"class_{idx}" for idx in range(len(class_names), FORCED_NUM_CLASSES)]
    else:
        class_names = FORCED_CLASS_NAMES

    # 预定义颜色
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 255, 0),
        (255, 128, 0),
        (128, 0, 255),
        (255, 0, 128),
    ]

    max_keypoints = 0
    for det in detections:
        if det.get("keypoints"):
            max_keypoints = max(max_keypoints, len(det["keypoints"]))
    keypoint_palette = _generate_color_palette(max_keypoints)

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        class_id = int(det.get("class_id", 0))
        class_id = max(0, min(class_id, FORCED_NUM_CLASSES - 1))
        det["class_id"] = class_id
        conf = det["confidence"]
        # 应用sigmoid变换得分
        conf_sigmoid = 1.0 / (1.0 + np.exp(-conf))

        # 选择颜色
        color = colors[class_id % len(colors)]

        # 绘制边界框
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        # 准备标签
        if class_names and 0 <= class_id < len(class_names):
            label = f"{class_names[class_id]}: {conf_sigmoid:.2f}"
        else:
            label = f"Class {class_id}: {conf_sigmoid:.2f}"

        # 绘制标签背景
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(vis_img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)

        # 绘制标签文本
        cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 绘制keypoints，根据是否被hm修正使用不同颜色
        keypoints = det.get("keypoints")
        initial_keypoints = det.get("initial_keypoints")

        vis_info_lines = []
        if keypoints:
            max_kps = len(keypoints)
            if initial_keypoints:
                initial_keypoints = initial_keypoints[:max_kps]
            visibilities_raw = det.get("keypoint_visibility")
            log_info(f"[vis debug] det #{i} raw visibilities: {visibilities_raw}", "detail")
            visibilities = []
            if visibilities_raw is not None:
                try:
                    arr = np.asarray(visibilities_raw, dtype=np.float32).reshape(-1)
                except Exception:
                    arr = np.array(visibilities_raw, dtype=np.float32).reshape(-1)
                visibilities = [float(v) for v in arr[:max_kps]]
            if len(visibilities) < max_kps:
                visibilities.extend([None] * (max_kps - len(visibilities)))
            log_info(f"[vis debug] det #{i} parsed visibilities: {visibilities}", "detail")
            for idx, (kp_x, kp_y) in enumerate(keypoints):
                vis_flag = visibilities[idx] if idx < len(visibilities) else None
                if vis_flag is not None:
                    visible = vis_flag > 0.5
                    kp_color = (0, 220, 0) if visible else (0, 0, 230)
                else:
                    visible = True
                    kp_color = keypoint_palette[idx % len(keypoint_palette)] if keypoint_palette else color

                # 检查该关键点是否被hm修正
                modified = False
                if initial_keypoints and idx < len(initial_keypoints):
                    init_x, init_y = initial_keypoints[idx]
                    # 计算位移距离
                    diff = np.sqrt((kp_x - init_x) ** 2 + (kp_y - init_y) ** 2)
                    if diff > 0.5:  # 位移超过0.5像素认为发生了修正
                        modified = True
                        # 绘制初始位置（灰色空心圆）
                        cv2.circle(vis_img, (int(init_x), int(init_y)), 5, (128, 128, 128), 1)
                        # 绘制连线显示修正方向（灰色虚线）
                        cv2.line(vis_img, (int(init_x), int(init_y)), (int(kp_x), int(kp_y)), (128, 128, 128), 1)

                # 绘制最终位置
                center_pt = (int(round(kp_x)), int(round(kp_y)))
                radius = 5
                cv2.circle(vis_img, center_pt, radius + 2, (255, 255, 255), 1)
                cv2.circle(vis_img, center_pt, radius, kp_color, -1)

                if not visible:
                    cv2.line(
                        vis_img,
                        (center_pt[0] - radius, center_pt[1] - radius),
                        (center_pt[0] + radius, center_pt[1] + radius),
                        (255, 255, 255),
                        1,
                    )
                    cv2.line(
                        vis_img,
                        (center_pt[0] - radius, center_pt[1] + radius),
                        (center_pt[0] + radius, center_pt[1] - radius),
                        (255, 255, 255),
                        1,
                    )

                if vis_flag is not None:
                    vis_label = 1 if vis_flag > 0.5 else 0
                    text = f"{idx}:{vis_label}"

                    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    img_h, img_w = vis_img.shape[:2]
                    text_x = center_pt[0] + radius + 4
                    text_y = center_pt[1] + radius
                    if text_y + baseline + 2 > img_h:
                        text_y = center_pt[1] - radius - 4
                        if text_y - text_size[1] - 2 < 0:
                            text_y = text_size[1] + 2
                    if text_x + text_size[0] + 2 > img_w:
                        text_x = max(center_pt[0] - radius - text_size[0] - 4, 0)
                    text_origin = (int(text_x), int(text_y))
                    bg_tl = (text_origin[0] - 2, text_origin[1] - text_size[1] - 2)
                    bg_br = (text_origin[0] + text_size[0] + 2, text_origin[1] + baseline + 2)
                    cv2.rectangle(vis_img, bg_tl, bg_br, (0, 0, 0), -1)
                    cv2.putText(
                        vis_img,
                        text,
                        text_origin,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    vis_info_lines.append(text)
                else:
                    vis_info_lines.append(f"{idx}:?")

            if keypoint_edges:
                for edge in keypoint_edges:
                    if len(edge) != 2:
                        continue
                    i, j = edge
                    if 0 <= i < len(keypoints) and 0 <= j < len(keypoints):
                        edge_color = keypoint_palette[i % len(keypoint_palette)] if keypoint_palette else color
                        pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                        pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                        cv2.line(vis_img, pt1, pt2, edge_color, 2)

    if save_path:
        cv2.imwrite(save_path, vis_img)
        log_info(f"Visualization saved to: {save_path}")

    return vis_img


def visualize_and_save_heatmaps(
    heatmaps,
    save_path_prefix,
    class_names=None,
    input_size=(512, 512),
    threshold=0.1,
    background=None,
    center=None,
    scale=None,
    overlay_alpha=0.4,
    max_maps=None,
):
    """保存热力图，将同一 head 的通道绘制在一个多子图图片中，并追加 combined."""

    import cv2
    import numpy as np
    import math

    heatmaps = np.asarray(heatmaps)
    if heatmaps.ndim == 4:
        heatmaps = heatmaps[0]
    elif heatmaps.ndim == 3 and heatmaps.shape[0] == 1:
        heatmaps = heatmaps[0]

    total_maps = heatmaps.shape[0]
    num_maps = total_maps if max_maps is None else min(total_maps, max_maps)
    heatmaps_sigmoid = 1 / (1 + np.exp(-heatmaps))

    log_info(f"保存类别热力图数量: {num_maps}")

    saved_heatmaps = []
    panels = []

    background_img = None
    inv_trans = None
    if background is not None and center is not None and scale is not None:
        background_img = background.copy()
        inv_trans = get_affine_transform(center, scale, 0, [input_size[0], input_size[1]], inv=1)

    for cls_idx in range(num_maps):
        heatmap = heatmaps_sigmoid[cls_idx]
        max_val = float(heatmap.max())
        if max_val < threshold:
            log_info(
                f"heatmap class {cls_idx} max {max_val:.4f} below threshold {threshold:.4f}, saving anyway",
                "warn",
            )

        heatmap_resized = cv2.resize(heatmap, input_size, interpolation=cv2.INTER_LINEAR)

        if inv_trans is not None:
            projected = cv2.warpAffine(
                heatmap_resized,
                inv_trans,
                (background_img.shape[1], background_img.shape[0]),
                flags=cv2.INTER_LINEAR,
            )
            heatmap_norm = np.clip(projected, 0.0, 1.0)
        else:
            heatmap_norm = np.clip(heatmap_resized, 0.0, 1.0)

        if class_names and cls_idx < len(class_names):
            label_name = class_names[cls_idx]
        elif class_names:
            label_name = class_names[-1]
        elif max_maps == FORCED_NUM_CLASSES:
            label_name = FORCED_CLASS_NAMES[min(cls_idx, FORCED_NUM_CLASSES - 1)]
        else:
            label_name = f"Class_{cls_idx}"

        panels.append((heatmap_norm, label_name, max_val))
        saved_heatmaps.append(
            {
                "class_id": cls_idx,
                "class_name": label_name,
                "max_activation": max_val,
                "save_path": None,
            }
        )

    if not panels:
        return saved_heatmaps

    combined = np.max(heatmaps_sigmoid[:num_maps], axis=0)
    combined_resized = cv2.resize(combined, input_size, interpolation=cv2.INTER_LINEAR)
    if inv_trans is not None:
        projected = cv2.warpAffine(
            combined_resized,
            inv_trans,
            (background_img.shape[1], background_img.shape[0]),
            flags=cv2.INTER_LINEAR,
        )
        combined_norm = np.clip(projected, 0.0, 1.0)
    else:
        combined_norm = np.clip(combined_resized, 0.0, 1.0)

    panels.append((combined_norm, "Combined", float(combined_norm.max())))

    panel_count = len(panels)
    cols = min(3, panel_count)
    rows = int(math.ceil(panel_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)

    for ax in axes.flat:
        ax.set_axis_off()

    for idx, (panel, title, max_val) in enumerate(panels):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        ax.set_axis_off()
        if background_img is not None:
            bg_rgb = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
            ax.imshow(bg_rgb)
            ax.imshow(panel, cmap="jet", vmin=0.0, vmax=1.0, alpha=overlay_alpha)
        else:
            ax.imshow(panel, cmap="jet", vmin=0.0, vmax=1.0)
        ax.set_title(f"{title} (max={max_val:.3f})")

    combined_path = f"{save_path_prefix}_heatmap_combined.png"
    fig.tight_layout()
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_info(f"综合热力图已保存: {combined_path}")

    for record in saved_heatmaps:
        record["save_path"] = combined_path

    return saved_heatmaps


def load_yolo_ground_truth(label_path, original_size, num_joints):
    gt = []
    width, height = original_size
    if not os.path.exists(label_path):
        log_warn(f"[ground_truth] label file not found: {label_path}")
        return gt
    try:
        with open(label_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except OSError as exc:
        log_error(f"[ground_truth] failed to read label file: {label_path} | error: {exc}")
        return gt

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cls_id = max(0, min(cls_id, FORCED_NUM_CLASSES - 1))
        xc, yc, w, h = map(float, parts[1:5])
        xc *= width
        yc *= height
        w *= width
        h *= height
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        entry = {
            "class_id": cls_id,
            "bbox": [x1, y1, x2, y2],
            "confidence": 1.0,
            "keypoints": [],
        }

        remaining = parts[5:]
        if num_joints > 0 and len(remaining) >= num_joints * 2:
            kps = []
            for j in range(num_joints):
                kx = float(remaining[2 * j])
                ky = float(remaining[2 * j + 1])
                if 0.0 <= kx <= 1.0:
                    kx *= width
                if 0.0 <= ky <= 1.0:
                    ky *= height
                kps.append((kx, ky))
            entry["keypoints"] = kps
        gt.append(entry)
    return gt


def _log_prediction_vs_gt(pred, gt_entries):
    if not gt_entries:
        return

    def _iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
        area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
        denom = area1 + area2 - inter
        return inter / denom if denom > 0 else 0.0

    ious = [_iou(pred["bbox"], gt["bbox"]) for gt in gt_entries]
    best_idx = int(np.argmax(ious)) if ious else -1
    if best_idx < 0:
        return
    gt = gt_entries[best_idx]
    # IoU 与最佳 GT 索引
    try:
        best_iou = float(ious[best_idx])
        log_info(f"[compare] best match: best_gt_index={best_idx}, IoU={best_iou:.3f}", "header")
    except Exception:
        pass
    # 计算bbox中心点和宽高
    pred_bbox = pred["bbox"]
    pred_left, pred_top, pred_right, pred_bottom = pred_bbox
    center_x_numerator = pred_left + pred_right
    center_y_numerator = pred_top + pred_bottom
    pred_center = (center_x_numerator / 2, center_y_numerator / 2)
    pred_width = pred_right - pred_left
    pred_height = pred_bottom - pred_top
    pred_wh = (pred_width, pred_height)
    gt_bbox = gt["bbox"]
    gt_center = ((gt_bbox[0] + gt_bbox[2]) / 2, (gt_bbox[1] + gt_bbox[3]) / 2)
    gt_wh = (gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1])

    log_info(
        (
            f"[compare] pred bbox raw: left={pred_left:.2f}, top={pred_top:.2f}, "
            f"right={pred_right:.2f}, bottom={pred_bottom:.2f}"
        ),
        "pred",
    )
    log_info(
        (
            "[compare] pred center calc: "
            f"({pred_left:.2f} + {pred_right:.2f}) / 2 -> {pred_center[0]:.2f}, "
            f"({pred_top:.2f} + {pred_bottom:.2f}) / 2 -> {pred_center[1]:.2f}"
        ),
        "detail",
    )
    log_info(
        (
            "[compare] pred size calc: "
            f"{pred_right:.2f} - {pred_left:.2f} -> {pred_width:.2f}, "
            f"{pred_bottom:.2f} - {pred_top:.2f} -> {pred_height:.2f}"
        ),
        "detail",
    )
    log_info(
        f"[compare] pred bbox_center: ({pred_center[0]:.2f}, {pred_center[1]:.2f}), wh: ({pred_wh[0]:.2f}, {pred_wh[1]:.2f})",
        "pred",
    )
    log_info(
        f"[compare] gt   bbox_center: ({gt_center[0]:.2f}, {gt_center[1]:.2f}), wh: ({gt_wh[0]:.2f}, {gt_wh[1]:.2f})",
        "gt",
    )

    if pred.get("keypoints") and gt.get("keypoints"):
        pairs = zip(pred["keypoints"], gt["keypoints"])
        deltas = [(round(pk[0] - gk[0], 3), round(pk[1] - gk[1], 3)) for pk, gk in pairs]
        log_info(f"[compare] pred keypoints: {pred['keypoints']}", "pred")
        log_info(f"[compare] gt   keypoints: {gt['keypoints']}", "gt")
        log_info(f"[compare] keypoint delta (pred - gt): {deltas}", "delta")
        # per-kp L2 errors summary (plain content)
        try:
            import math

            l2_list = [math.sqrt(dx * dx + dy * dy) for dx, dy in deltas]
            log_info(f"[compare] keypoint L2 errors: {[round(v, 3) for v in l2_list]}", "detail")
        except Exception:
            pass
        # warn/info if any keypoint error is large/notable
        try:
            import math

            max_l2 = -1.0
            max_idx = -1
            for i, (dx, dy) in enumerate(deltas):
                l2 = math.sqrt(dx * dx + dy * dy)
                if l2 > max_l2:
                    max_l2, max_idx = l2, i
            import os as _os

            TH_INFO = float(_os.environ.get("KP_INFO_L2", 10.0))
            TH_WARN = float(_os.environ.get("KP_WARN_L2", 30.0))
            if max_l2 >= TH_WARN:
                log_warn(f"[compare] large keypoint error: max_L2={max_l2:.2f} at idx={max_idx}")
            elif max_l2 >= TH_INFO:
                log_info(f"[compare] notable keypoint error: max_L2={max_l2:.2f} at idx={max_idx}")
        except Exception:
            pass


def test_image_inference(pytorch_model, opt):
    """测试单张图像推理"""
    import onnxruntime as ort
    import numpy as np

    if not os.path.exists(opt.test_image):
        log_error(f"Test image not found: {opt.test_image}")
        return

    log_info("Loading ONNX model for inference...")
    ort_session = ort.InferenceSession(opt.output_path)
    output_names = [output.name for output in ort_session.get_outputs()]

    # 获取类别名称
    class_names = None
    if opt.dataset == "coco":
        class_names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
    if class_names:
        class_names = list(class_names)[:FORCED_NUM_CLASSES]
        if len(class_names) < FORCED_NUM_CLASSES:
            class_names += [f"class_{idx}" for idx in range(len(class_names), FORCED_NUM_CLASSES)]
    else:
        class_names = FORCED_CLASS_NAMES

    # 预处理图像
    log_info("Preprocessing image...")
    # 使用opt中的mean和std值（BGR格式）
    mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    log_info(f"mean / std: mean={mean.tolist()}, std={std.tolist()}")
    image_base_name = os.path.splitext(os.path.basename(opt.test_image))[0]
    (
        input_tensor,
        original_img,
        original_size,
        center,
        scale,
    ) = preprocess_image(
        opt.test_image,
        opt.input_h,
        opt.input_w,
        mean,
        std,
        use_rgb=getattr(opt, "input_rgb", False),
    )

    if getattr(opt, "network_input_txt", ""):
        try:
            loaded_tensor = load_network_input_txt(opt.network_input_txt, expected_shape=input_tensor.shape)
            diff = np.abs(loaded_tensor.astype(np.float32) - input_tensor)
            max_diff = float(diff.max())
            min_diff = float(diff.min())
            avg_diff = float(diff.mean())

            log_info(
                f"network_input_txt diff vs preprocessed tensor: max={max_diff:.6f}, min={min_diff:.6f}, avg={avg_diff:.6f}",
                "detail" if max_diff < 1e-3 else "warn",
            )

            if max_diff < 1e-6 or opt.force_network_input:
                input_tensor = loaded_tensor.astype(np.float32, copy=False)
                if not opt.force_network_input and max_diff >= 1e-6:
                    log_warn("Using provided network_input_txt after warning because difference is small but non-zero")
            else:
                log_warn(
                    "Loaded network_input_txt differs significantly from freshly preprocessed tensor; ignoring it."
                )
        except Exception as exc:
            log_error(f"Failed to load network input txt: {exc}")

    # ONNX推理
    log_info("Running ONNX inference...")
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outputs = ort_session.run(None, ort_inputs)

    # 保存热力图可视化结果
    if opt.save_vis and opt.save_heatmaps:
        heatmap_prefix = f"{image_base_name}"

        head_names = list(opt.heads.keys())
        if "hm" in head_names:
            hm_idx = head_names.index("hm")
            heatmaps = ort_outputs[hm_idx]
        else:
            heatmaps = ort_outputs[0]

        log_info("Saving heatmap visualizations...")
        saved_heatmaps = visualize_and_save_heatmaps(
            heatmaps,
            heatmap_prefix,
            class_names=class_names,
            input_size=(opt.input_w, opt.input_h),
            threshold=opt.heatmap_threshold,
            background=None,
            center=None,
            scale=None,
            max_maps=FORCED_NUM_CLASSES,
        )
        print("========> 没有加sigmoid之前的最大值", heatmaps.shape, heatmaps.max())
        log_info(f"Saved heatmap visualizations: {len(saved_heatmaps)}")

        if opt.task == "multi_pose" and "hm_hp" in head_names:
            hm_hp_idx = head_names.index("hm_hp")
            kp_heatmaps = ort_outputs[hm_hp_idx]
            if kp_heatmaps is not None:
                num_joints = kp_heatmaps.shape[1] if kp_heatmaps.ndim == 4 else kp_heatmaps.shape[0]
                kp_names = [f"joint_{i}" for i in range(num_joints)]
                kp_saved = visualize_and_save_heatmaps(
                    kp_heatmaps,
                    f"{heatmap_prefix}_kp",
                    class_names=kp_names,
                    input_size=(opt.input_w, opt.input_h),
                    threshold=opt.heatmap_threshold,
                    background=None,
                    center=None,
                    scale=None,
                )
                log_info(f"Saved keypoint heatmap visualizations: {len(kp_saved)}")

    # 后处理
    log_info("Postprocessing results...")
    detections = postprocess_detections(ort_outputs, opt, center, scale)

    save_onnx_io_dump(image_base_name, input_tensor, output_names, ort_outputs, detections)

    if opt.test_label:
        gt_entries = load_yolo_ground_truth(opt.test_label, original_size, getattr(opt, "num_joints", 0))
        log_info(f"[ground_truth] entries loaded: path={opt.test_label}, count={len(gt_entries)}")
        for idx, gt in enumerate(gt_entries):
            log_info(
                "  GT #{} class {} score {:.3f} bbox {}".format(
                    idx, gt["class_id"], gt.get("confidence", 1.0), gt["bbox"]
                ),
                "gt",
            )
            if gt.get("keypoints"):
                log_info(f"    keypoints: {gt['keypoints']}", "gt")

    log_info(f"Found detections: {len(detections)}")

    # 可视化结果
    if detections:
        log_info("Visualizing results...")

        # 保存路径
        if opt.save_vis:
            base_name = os.path.splitext(os.path.basename(opt.test_image))[0]
            save_path = f"{base_name}_detection_result.jpg"
        else:
            save_path = None

        vis_img = visualize_detections(
            original_img,
            detections,
            class_names,
            save_path,
            getattr(opt, "keypoint_edges", None),
        )

        # 显示检测信息
        for det in detections[:10]:  # 显示前10个检测结果
            class_name = (
                class_names[det["class_id"]]
                if class_names and det["class_id"] < len(class_names)
                else f"Class {det['class_id']}"
            )
            # 应用sigmoid变换得分
            score_sigmoid = 1.0 / (1.0 + np.exp(-det["confidence"]))
            log_info(
                f'  {class_name}: {score_sigmoid:.3f} at [{det["bbox"][0]:.1f}, {det["bbox"][1]:.1f}, {det["bbox"][2]:.1f}, {det["bbox"][3]:.1f}]',
                "pred",
            )
            if det.get("keypoints"):
                log_info(f"    keypoints: {det['keypoints']}", "pred")
            if opt.test_label and det.get("keypoints"):
                _log_prediction_vs_gt(det, gt_entries)
    else:
        log_info("No detections found")


def main():
    # Create opts object and add ONNX specific arguments
    opts_obj = opts()
    opts_obj.parser.add_argument("--output_path", default="", help="Output ONNX file path")
    opts_obj.parser.add_argument("--opset_version", type=int, default=11, help="ONNX opset version")
    opts_obj.parser.add_argument("--verify", action="store_true", help="Verify the converted ONNX model")
    opts_obj.parser.add_argument("--test_image", default="", help="Path to test image for inference visualization")
    opts_obj.parser.add_argument("--save_vis", action="store_true", help="Save visualization results")
    opts_obj.parser.add_argument("--save_heatmaps", action="store_true", help="Save heatmap visualizations")
    opts_obj.parser.add_argument(
        "--heatmap_threshold", type=float, default=0.1, help="Minimum activation threshold for saving heatmaps"
    )
    opts_obj.parser.add_argument("--test_label", default="", help="Path to YOLO-format label file for the test image")
    opts_obj.parser.add_argument(
        "--network_input_txt",
        default="",
        help="Path to pre-saved *_network_input.txt file (expects normalized CHW data)",
    )
    opts_obj.parser.add_argument(
        "--force_network_input",
        action="store_true",
        help="Use the provided network_input_txt even if it differs from fresh preprocessing",
    )

    opt = opts_obj.parse()

    # Set defaults for new arguments if not present
    if not hasattr(opt, "output_path"):
        opt.output_path = ""
    if not hasattr(opt, "opset_version"):
        opt.opset_version = 11
    if not hasattr(opt, "verify"):
        opt.verify = False
    if not hasattr(opt, "test_image"):
        opt.test_image = ""
    if not hasattr(opt, "save_vis"):
        opt.save_vis = False
    if not hasattr(opt, "save_heatmaps"):
        opt.save_heatmaps = False
    if not hasattr(opt, "heatmap_threshold"):
        opt.heatmap_threshold = 0.1
    if not hasattr(opt, "dynamic_batch"):
        opt.dynamic_batch = False
    if not hasattr(opt, "test_label"):
        opt.test_label = ""
    if not hasattr(opt, "network_input_txt"):
        opt.network_input_txt = ""
    if not hasattr(opt, "force_network_input"):
        opt.force_network_input = False
    if not hasattr(opt, "input_rgb"):
        opt.input_rgb = False

    # Set default dataset info if not specified
    if not hasattr(opt, "heads") or opt.heads is None:
        from datasets.dataset_factory import get_dataset

        Dataset = get_dataset(opt.dataset, opt.task)
        opt = opts_obj.update_dataset_info_and_set_heads(opt, Dataset)

    convert_to_onnx(opt)


if __name__ == "__main__":
    main()
