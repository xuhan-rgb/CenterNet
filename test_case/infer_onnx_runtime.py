#!/usr/bin/env python3
"""
ONNX Runtime 推理脚本
----------------------
直接载入 ONNX 模型，对单张图片执行推理并完成后处理，无需额外的数据集配置。

特性：
- 自动解析模型输入尺寸与输出头信息
- 根据输出头自动判定任务类型（ctdet / multi_pose），也可手动指定
- 可选保存可视化、JSON 结果，并支持跨类别 NMS
- 默认采用均值 0 / 方差 1 归一化，可通过参数覆盖
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import math

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required to load YAML config files.") from exc

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise ImportError("onnxruntime is required to run this script.") from exc

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DETAIL_LOG_ENABLED = bool(int(os.environ.get("ONNX_INFER_DETAIL_LOGS", "0")))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def log_info(msg, detail=False):
    if detail and not DETAIL_LOG_ENABLED:
        return
    print(msg)


def log_warn(msg):
    print(msg)


def log_error(msg):
    print(msg)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _normalize_class_names(class_names, num_classes=None):
    names = list(class_names) if class_names else []
    if num_classes is None:
        return names
    num_classes = max(0, int(num_classes))
    if len(names) < num_classes:
        names.extend(f"class_{idx}" for idx in range(len(names), num_classes))
    else:
        names = names[:num_classes]
    return names


def load_yaml_config(config_path):
    abs_path = config_path
    if not os.path.isabs(abs_path):
        if os.path.exists(config_path):
            abs_path = os.path.abspath(config_path)
        else:
            abs_path = os.path.join(_PROJECT_ROOT, config_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")
    with open(abs_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data, abs_path


def resolve_relative_path(path_value, search_dirs, must_exist=False):
    """
    Resolve a potentially relative path against a list of base directories.

    When must_exist is True, the function returns the first existing candidate.
    Otherwise it joins with the first search directory for deterministic output.
    """
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    search_dirs = [base for base in search_dirs if base]
    if must_exist:
        for base_dir in search_dirs:
            candidate = os.path.join(base_dir, path_value)
            if os.path.exists(candidate):
                return candidate
        primary_base = search_dirs[0] if search_dirs else os.getcwd()
        return os.path.abspath(os.path.join(primary_base, path_value))
    primary_base = search_dirs[0] if search_dirs else os.getcwd()
    return os.path.abspath(os.path.join(primary_base, path_value))


def _transpose_and_gather_feat(feat, ind):
    batch = feat.shape[0]
    feat = feat.transpose(0, 2, 3, 1).reshape(batch, -1, feat.shape[1])
    ind = ind.astype(np.int64)
    batch_idx = np.arange(batch)[:, None]
    return feat[batch_idx, ind, :]


def _nms(heat, kernel=3):
    if heat.size == 0:
        return heat
    batch, channel, height, width = heat.shape
    kernel_mat = np.ones((kernel, kernel), dtype=heat.dtype)
    pooled = np.empty_like(heat)
    for b in range(batch):
        for c in range(channel):
            pooled[b, c] = cv2.dilate(heat[b, c], kernel_mat)
    keep = (heat == pooled).astype(np.float32)
    return heat * keep


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.shape
    flat = scores.reshape(batch, cat, -1)
    K = min(K, flat.shape[2])
    idx = np.argpartition(-flat, K - 1, axis=2)[:, :, :K]
    topk_scores = np.take_along_axis(flat, idx, axis=2)
    order = np.argsort(-topk_scores, axis=2)
    idx = np.take_along_axis(idx, order, axis=2)
    topk_scores = np.take_along_axis(topk_scores, order, axis=2)
    topk_inds = idx % (height * width)
    topk_ys = (topk_inds // width).astype(np.float32)
    topk_xs = (topk_inds % width).astype(np.float32)
    return topk_scores.astype(np.float32), topk_inds.astype(np.int64), topk_ys, topk_xs


def _topk(scores, K=40):
    batch, cat, height, width = scores.shape
    flat = scores.reshape(batch, -1)
    total = flat.shape[1]
    K = min(K, total)
    idx = np.argpartition(-flat, K - 1, axis=1)[:, :K]
    topk_score = np.take_along_axis(flat, idx, axis=1)
    order = np.argsort(-topk_score, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    topk_score = np.take_along_axis(topk_score, order, axis=1)
    spatial = height * width
    topk_clses = (idx // spatial).astype(np.float32)
    topk_inds = (idx % spatial).astype(np.int64)
    topk_ys = (topk_inds // width).astype(np.float32)
    topk_xs = (topk_inds % width).astype(np.float32)
    return topk_score.astype(np.float32), topk_inds, topk_clses, topk_ys, topk_xs


def get_dir(src_point, rot_rad):
    sn, cs = math.sin(rot_rad), math.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, scale, rot, output_size, shift=None, inv=0):
    if shift is None:
        shift = np.array([0, 0], dtype=np.float32)
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = np.array(get_dir([0, src_w * -0.5], rot_rad), dtype=np.float32)
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


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape, dtype=np.float32)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def preprocess_image(image_path, input_h, input_w, mean, std, use_rgb=False):
    """CenterNet 标准预处理流程"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    log_info(
        f"Input image shape / size: shape={img_bgr.shape}, input_h={input_h}, input_w={input_w}",
        detail=True,
    )

    original_img = img_bgr.copy()
    if use_rgb:
        img_proc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_proc = img_bgr

    height, width = img_proc.shape[:2]
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    scale = max(height, width) * 1.0

    trans_input = get_affine_transform(center, scale, 0, [input_w, input_h])
    inp = cv2.warpAffine(img_proc, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
    inp = inp.astype(np.float32) / 255.0

    inp = (inp - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    inp = inp.transpose(2, 0, 1)
    inp = inp.reshape(1, inp.shape[0], inp.shape[1], inp.shape[2])

    return inp, original_img, (height, width), center, scale


def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.shape
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    center_x = xs[..., None]
    center_y = ys[..., None]
    if reg is not None:
        reg_vals = _transpose_and_gather_feat(reg, inds).reshape(batch, K, 2)
        center_x = center_x + reg_vals[..., 0:1]
        center_y = center_y + reg_vals[..., 1:2]
    else:
        center_x = center_x + 0.5
        center_y = center_y + 0.5

    wh_vals = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh_vals = wh_vals.reshape(batch, K, cat, 2)
        cls_idx = clses.astype(np.int64)
        batch_idx = np.arange(batch)[:, None]
        det_idx = np.arange(wh_vals.shape[1])[None, :]
        wh_vals = wh_vals[batch_idx, det_idx, cls_idx, :]
    else:
        wh_vals = wh_vals.reshape(batch, K, 2)

    half_w = wh_vals[..., 0:1] / 2.0
    half_h = wh_vals[..., 1:2] / 2.0
    bboxes = np.concatenate(
        [
            center_x - half_w,
            center_y - half_h,
            center_x + half_w,
            center_y + half_h,
        ],
        axis=2,
    )

    detections = np.concatenate([bboxes, scores[..., None], clses[..., None]], axis=2).astype(np.float32)
    return detections


def multi_pose_decode(
    heat,
    wh,
    kps,
    reg=None,
    hm_hp=None,
    hp_offset=None,
    hp_vis=None,
    K=100,
    return_initial_kps=False,
):
    batch, cat, height, width = heat.shape
    num_joints = kps.shape[1] // 2 if kps.size > 0 else 0
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    base_xs = xs.astype(np.float32)
    base_ys = ys.astype(np.float32)

    if num_joints > 0:
        gathered_kps = _transpose_and_gather_feat(kps, inds).reshape(batch, K, num_joints * 2)
        kps_adjusted = gathered_kps.copy()
        kps_adjusted[..., ::2] += base_xs[..., None]
        kps_adjusted[..., 1::2] += base_ys[..., None]

        xs_int = base_xs.copy()
        ys_int = base_ys.copy()

        initial_kps = gathered_kps.reshape(batch, K, num_joints, 2)
        initial_kps[..., 0] += xs_int[..., None]
        initial_kps[..., 1] += ys_int[..., None]
        initial_kps_flat = initial_kps.reshape(batch, K, num_joints * 2)
    else:
        kps_adjusted = np.zeros((batch, K, 0), dtype=np.float32)
        initial_kps_flat = np.zeros((batch, K, 0), dtype=np.float32)

    center_x = base_xs[..., None]
    center_y = base_ys[..., None]
    if reg is not None:
        reg_vals = _transpose_and_gather_feat(reg, inds).reshape(batch, K, 2)
        center_x = center_x + reg_vals[..., 0:1]
        center_y = center_y + reg_vals[..., 1:2]
    else:
        center_x = center_x + 0.5
        center_y = center_y + 0.5

    wh_vals = _transpose_and_gather_feat(wh, inds).reshape(batch, K, 2)
    half_w = wh_vals[..., 0:1] / 2.0
    half_h = wh_vals[..., 1:2] / 2.0
    bboxes = np.concatenate(
        [
            center_x - half_w,
            center_y - half_h,
            center_x + half_w,
            center_y + half_h,
        ],
        axis=2,
    )

    hp_vis_scores = None
    if hp_vis is not None and num_joints > 0:
        hp_vis_vals = _transpose_and_gather_feat(hp_vis, inds).reshape(batch, K, num_joints)
        hp_vis_scores = _sigmoid(hp_vis_vals)

    if hm_hp is not None and num_joints > 0:
        hm_hp = _nms(hm_hp)
        thresh = 0.1

        reg_kps = kps_adjusted.reshape(batch, K, num_joints, 2).transpose(0, 2, 1, 3)
        reg_kps_exp = np.repeat(reg_kps[..., None, :], K, axis=3)

        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)
        if hp_offset is not None:
            offset_vals = _transpose_and_gather_feat(hp_offset, hm_inds.reshape(batch, -1))
            offset_vals = offset_vals.reshape(batch, num_joints, K, 2)
            hm_xs = hm_xs + offset_vals[..., 0]
            hm_ys = hm_ys + offset_vals[..., 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        topk_mask = (hm_score > thresh).astype(np.float32)
        hm_score = (1 - topk_mask) * -1 + topk_mask * hm_score
        hm_ys = (1 - topk_mask) * -10000 + topk_mask * hm_ys
        hm_xs = (1 - topk_mask) * -10000 + topk_mask * hm_xs

        hm_candidates = np.stack([hm_xs, hm_ys], axis=-1)
        hm_kps_expanded = np.repeat(hm_candidates[:, :, None, :, :], K, axis=2)

        dist = np.sqrt(((reg_kps_exp - hm_kps_expanded) ** 2).sum(axis=4))
        selected_idx = np.argmin(dist, axis=3)
        selected_scores = np.take_along_axis(hm_score, selected_idx, axis=2)[..., None]

        idx_expanded = np.repeat(selected_idx[..., None], 2, axis=3)
        hm_kps_selected = np.take_along_axis(hm_candidates, idx_expanded, axis=2)

        l = bboxes[:, :, 0][:, None, :, None]
        t = bboxes[:, :, 1][:, None, :, None]
        r = bboxes[:, :, 2][:, None, :, None]
        b = bboxes[:, :, 3][:, None, :, None]

        tb_margin_thresh = (b - t) / 4.0
        lr_margin_thresh = (r - l) / 4.0
        size_term = np.maximum(b - t, r - l) * 0.3

        cond_left = hm_kps_selected[..., 0:1] < (l - lr_margin_thresh)
        cond_right = hm_kps_selected[..., 0:1] > (r + lr_margin_thresh)
        cond_top = hm_kps_selected[..., 1:2] < (t - tb_margin_thresh)
        cond_bottom = hm_kps_selected[..., 1:2] > (b + tb_margin_thresh)
        cond_score = selected_scores < thresh
        selected_dist = np.take_along_axis(dist, selected_idx[..., None], axis=3)
        cond_dist = selected_dist > size_term

        reject_mask = (cond_left | cond_right | cond_top | cond_bottom | cond_score | cond_dist).astype(np.float32)
        reject_mask = np.repeat(reject_mask, 2, axis=3)

        mixed_kps = (1 - reject_mask) * hm_kps_selected + reject_mask * reg_kps
        kps_adjusted = mixed_kps.transpose(0, 2, 1, 3).reshape(batch, K, num_joints * 2)

    pieces = [bboxes, scores[..., None], kps_adjusted]
    if hp_vis_scores is not None:
        pieces.append(hp_vis_scores)
    pieces.append(clses[..., None])
    detections = np.concatenate(pieces, axis=2).astype(np.float32)

    if return_initial_kps:
        return detections, initial_kps_flat
    return detections


def ctdet_post_process(dets, c, s, h, w, num_classes):
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = classes == j
            top_preds[j + 1] = np.concatenate(
                [dets[i, inds, :4].astype(np.float32), dets[i, inds, 4:5].astype(np.float32)], axis=1
            ).tolist()
        ret.append(top_preds)
    return ret


def multi_pose_post_process(dets, c, s, h, w, num_joints=None):
    ret = []
    B, max_dets, dim = dets.shape
    inferred_joints = num_joints
    if inferred_joints is None:
        if dim < 6:
            inferred_joints = 0
        else:
            remaining = dim - 6
            if remaining <= 0:
                inferred_joints = 0
            elif remaining % 3 == 0:
                inferred_joints = remaining // 3
            elif remaining % 2 == 0:
                inferred_joints = remaining // 2
            else:
                inferred_joints = remaining // 2

    keypoint_len = inferred_joints * 2
    keypoint_end = 5 + keypoint_len
    vis_len = 0
    vis_start = keypoint_end
    vis_end = vis_start
    if inferred_joints > 0 and dim >= keypoint_end + inferred_joints + 1:
        vis_len = inferred_joints
        vis_end = vis_start + vis_len

    for i in range(B):
        classes = dets[i, :, -1].astype(np.int32)
        bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h)).reshape(-1, 4)
        score = dets[i, :, 4:5]
        pieces = [bbox, score]

        if inferred_joints > 0 and keypoint_end <= dim:
            pts = transform_preds(dets[i, :, 5:keypoint_end].reshape(-1, 2), c[i], s[i], (w, h)).reshape(
                -1, keypoint_len
            )
            pieces.append(pts)
        if vis_len > 0 and vis_end <= dim:
            pieces.append(dets[i, :, vis_start:vis_end])

        combined = np.concatenate(pieces, axis=1).astype(np.float32)

        top_preds = {}
        unique_classes = np.unique(classes)
        for cls_idx in unique_classes:
            if cls_idx < 0:
                continue
            cls_mask = classes == cls_idx
            if not np.any(cls_mask):
                continue
            cls_key = int(cls_idx) + 1
            top_preds[cls_key] = combined[cls_mask].tolist()
        ret.append(top_preds)
    return ret


def _bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def _apply_cross_class_nms(detections, iou_thresh):
    if iou_thresh is None or iou_thresh <= 0 or len(detections) <= 1:
        return detections

    detections = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
    keep = []
    suppressed = [False] * len(detections)

    for i, det in enumerate(detections):
        if suppressed[i]:
            continue
        keep.append(det)
        bbox_i = det.get("bbox", [0, 0, 0, 0])
        for j in range(i + 1, len(detections)):
            if suppressed[j]:
                continue
            bbox_j = detections[j].get("bbox", [0, 0, 0, 0])
            if _bbox_iou(bbox_i, bbox_j) > iou_thresh:
                suppressed[j] = True
    return keep


def postprocess_detections(outputs, opt, center, scale):
    head_names = list(opt.heads.keys())
    head_map = {name: outputs[idx] for idx, name in enumerate(head_names)}

    def _to_array(name):
        if name not in head_map:
            return None
        arr = np.asarray(head_map[name], dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[None, ...]
        return arr

    hm = _to_array("hm")
    if hm is None:
        raise ValueError("ONNX outputs missing 'hm' head")
    hm = _sigmoid(hm)

    num_classes = getattr(opt, "num_classes", None)
    if num_classes is None or int(num_classes) <= 0:
        num_classes = hm.shape[1] if hm.ndim >= 3 else 1
    num_classes = max(1, int(num_classes))

    score_thresh = float(getattr(opt, "vis_thresh", 0.3))
    results = []

    if opt.task == "ctdet":
        wh = _to_array("wh")
        if wh is None:
            raise ValueError("ONNX outputs missing 'wh' head for ctdet")
        reg = _to_array("reg") if "reg" in head_map else None
        cat_spec_wh = bool(getattr(opt, "cat_spec_wh", False))

        dets = ctdet_decode(
            hm,
            wh,
            reg=reg,
            cat_spec_wh=cat_spec_wh,
            K=int(getattr(opt, "K", 100)),
        )
        dets = dets.reshape(1, -1, dets.shape[2])
        processed = ctdet_post_process(
            dets.copy(),
            [center],
            [scale],
            hm.shape[3],
            hm.shape[2],
            num_classes,
        )

        for cls_id, bboxes in processed[0].items():
            for bbox in bboxes:
                if len(bbox) < 5 or bbox[4] < score_thresh:
                    continue
                class_id = int(cls_id - 1)
                class_id = max(0, min(class_id, num_classes - 1))
                results.append(
                    {
                        "bbox": [float(b) for b in bbox[:4]],
                        "confidence": float(bbox[4]),
                        "class_id": class_id,
                    }
                )

    elif opt.task == "multi_pose":
        wh = _to_array("wh")
        hps = _to_array("hps")
        if wh is None or hps is None:
            raise ValueError("ONNX outputs missing multi-pose heads 'wh' or 'hps'")
        reg = _to_array("reg") if "reg" in head_map else None
        hm_hp = _to_array("hm_hp") if "hm_hp" in head_map else None
        hp_offset = _to_array("hp_offset") if "hp_offset" in head_map else None
        hp_vis = _to_array("hp_vis") if "hp_vis" in head_map else None

        if hm_hp is not None:
            hm_hp = _sigmoid(hm_hp)

        dets = multi_pose_decode(
            hm,
            wh,
            hps,
            reg=reg if getattr(opt, "reg_offset", False) else None,
            hm_hp=hm_hp if getattr(opt, "hm_hp", False) else None,
            hp_offset=hp_offset if getattr(opt, "reg_hp_offset", False) else None,
            hp_vis=hp_vis,
            K=int(getattr(opt, "K", 100)),
        )

        raw_dets = dets.copy()
        dets = raw_dets.reshape(1, -1, raw_dets.shape[2])

        processed = multi_pose_post_process(
            dets.copy(),
            [center],
            [scale],
            hm.shape[2],
            hm.shape[3],
            getattr(opt, "num_joints", None),
        )

        num_joints = int(getattr(opt, "num_joints", 0))

        for cls_id, entries in processed[0].items():
            for entry_idx, entry in enumerate(entries):
                if len(entry) < 5 or entry[4] < score_thresh:
                    continue
                bbox = [float(b) for b in entry[:4]]
                keypoints = []
                visibilities = []
                if num_joints > 0:
                    kp_vals = entry[5 : 5 + num_joints * 2]
                    keypoints = [(float(kp_vals[2 * j]), float(kp_vals[2 * j + 1])) for j in range(num_joints)]
                    vis_start = 5 + num_joints * 2
                    vis_end = vis_start + num_joints
                    if len(entry) >= vis_end:
                        visibilities = [float(entry[vis_start + j]) for j in range(num_joints)]
                class_id = int(cls_id - 1)
                class_id = max(0, min(class_id, num_classes - 1))
                result_dict = {
                    "bbox": bbox,
                    "confidence": float(entry[4]),
                    "class_id": class_id,
                    "keypoints": keypoints,
                }
                if visibilities:
                    result_dict["keypoint_visibility"] = visibilities
                results.append(result_dict)
    else:
        raise ValueError(f"Unsupported task '{opt.task}' for post-processing")

    cross_nms_thresh = float(getattr(opt, "cross_class_nms_thresh", -1.0))
    if cross_nms_thresh > 0:
        before_count = len(results)
        results = _apply_cross_class_nms(results, cross_nms_thresh)
        log_info(
            f"[cross-nms] merged detections with IoU>{cross_nms_thresh:.2f}: {before_count} -> {len(results)}",
            detail=True,
        )

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results[: int(getattr(opt, "max_dets", 50))]


def visualize_detections(image, detections, class_names=None, save_path=None, keypoint_edges=None):
    vis_img = image.copy()

    max_cls_id = max((int(det.get("class_id", -1)) for det in detections), default=-1)
    inferred_num = max_cls_id + 1 if max_cls_id >= 0 else 0

    if class_names:
        class_names = _normalize_class_names(class_names, max(inferred_num, len(class_names)))
    else:
        target_num = inferred_num if inferred_num > 0 else 0
        if target_num <= 0:
            target_num = 1
        class_names = [f"class_{idx}" for idx in range(target_num)]

    num_classes = len(class_names)

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

    kp_colors = [
        (255, 255, 255),
        (0, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ]

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls_id = int(det.get("class_id", -1))
        cls_id = max(0, min(cls_id, num_classes - 1))
        label = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        score = det.get("confidence", 0.0)
        color = colors[cls_id % len(colors)]

        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {score:.2f}"
        cv2.putText(vis_img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        keypoints = det.get("keypoints", [])
        keypoint_vis = det.get("keypoint_visibility", [])
        for idx, kp in enumerate(keypoints):
            kp_color = kp_colors[idx % len(kp_colors)]
            kx, ky = int(kp[0]), int(kp[1])
            cv2.circle(vis_img, (kx, ky), 3, kp_color, -1)
            if keypoint_vis and idx < len(keypoint_vis):
                vis_value = keypoint_vis[idx]
                cv2.putText(vis_img, f"{vis_value:.1f}", (kx + 2, ky - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, kp_color, 1)

        if keypoint_edges:
            for edge in keypoint_edges:
                if len(edge) < 2:
                    continue
                p1, p2 = edge[:2]
                if p1 < len(keypoints) and p2 < len(keypoints):
                    pt1 = keypoints[p1]
                    pt2 = keypoints[p2]
                    cv2.line(vis_img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 1)

    if save_path:
        cv2.imwrite(save_path, vis_img)
        cv2.imshow("vis", vis_img)
        cv2.waitKey(0)
    return vis_img


def _parse_triplet(raw_value, default):
    if raw_value is None or raw_value == "":
        values = list(default)
    elif isinstance(raw_value, (list, tuple)):
        if len(raw_value) != 3:
            raise ValueError(f"Expected three values but got: {raw_value}")
        values = [float(v) for v in raw_value]
    else:
        tokens = [tok for tok in str(raw_value).replace(",", " ").split() if tok]
        if len(tokens) != 3:
            raise ValueError(f"Expected three values but got: {raw_value}")
        values = [float(tok) for tok in tokens]
    arr = np.array(values, dtype=np.float32).reshape(1, 1, 3)
    return arr


def _load_class_names(path):
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"class name file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    return names or None


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _infer_task(args_task, head_names):
    if args_task != "auto":
        return args_task
    pose_heads = {"hps", "hm_hp", "hp_offset", "hp_vis"}
    if pose_heads.intersection(head_names):
        return "multi_pose"
    return "ctdet"


def _determine_input_size(meta_shape, user_height, user_width):
    def _resolve(dim_value, user_value, name):
        if user_value and user_value > 0:
            return int(user_value)
        if isinstance(dim_value, (int, np.integer)) and int(dim_value) > 0:
            return int(dim_value)
        if dim_value in (None, "None"):
            raise ValueError(f"Model input dimension '{name}' is dynamic, please specify --input_{name}.")
        try:
            dim_int = int(dim_value)
            if dim_int > 0:
                return dim_int
        except Exception:
            pass
        raise ValueError(f"Unable to determine input {name}. Please specify --input_{name}.")

    if len(meta_shape) != 4:
        raise ValueError(f"Expected 4D input tensor [N,C,H,W], got shape={meta_shape}")
    _, _, h_dim, w_dim = meta_shape
    input_h = _resolve(h_dim, user_height, "h")
    input_w = _resolve(w_dim, user_width, "w")
    return input_h, input_w


def _create_opt_namespace(
    cfg,
    ort_outputs,
    output_names,
    original_image,
    input_h,
    input_w,
    class_names,
):
    heads = {}
    for name, output in zip(output_names, ort_outputs):
        channels = output.shape[1] if (output.ndim >= 2) else output.shape[-1]
        heads[name] = int(channels)

    task = _infer_task(getattr(cfg, "task", "auto"), set(heads.keys()))

    cfg_num_classes = getattr(cfg, "num_classes", -1)
    if cfg_num_classes is None:
        cfg_num_classes = -1
    num_classes = int(cfg_num_classes)
    if num_classes <= 0:
        if "hm" in heads and heads["hm"] > 0:
            num_classes = heads["hm"]
        else:
            first = ort_outputs[0]
            num_classes = int(first.shape[1]) if first.ndim >= 2 else 1
    num_classes = max(1, num_classes)

    cfg_num_joints = getattr(cfg, "num_joints", -1)
    num_joints = int(cfg_num_joints)
    if num_joints < 0 and "hps" in heads:
        hps_channels = heads["hps"]
        if hps_channels % 2 == 0:
            num_joints = hps_channels // 2
        else:
            log_warn(f"[infer] unexpected hps channel count {hps_channels}, fallback num_joints=0")
            num_joints = 0
    num_joints = max(0, num_joints)

    opt = SimpleNamespace()
    opt.task = task
    opt.heads = heads
    opt.num_classes = num_classes
    opt.num_joints = num_joints
    opt.K = int(getattr(cfg, "topk", 100))
    opt.vis_thresh = float(getattr(cfg, "vis_thresh", 0.3))
    opt.cross_class_nms_thresh = float(getattr(cfg, "cross_class_nms_thresh", -1.0))
    opt.hm_hp = "hm_hp" in heads
    opt.reg_offset = "reg" in heads
    opt.reg_hp_offset = "hp_offset" in heads
    opt.class_names = class_names
    opt.original_image = original_image
    opt.output_dir = os.path.abspath(getattr(cfg, "output_dir", "result/onnx_infer"))
    opt.save_vis = bool(getattr(cfg, "save_vis", False))
    opt.save_json = bool(getattr(cfg, "save_json", False))
    opt.print_results = bool(getattr(cfg, "print_results", False))
    opt.max_dets = int(getattr(cfg, "max_dets", 50))
    return opt


def _save_results(opt, image_name, detections, class_names):
    if opt.save_json:
        json_path = os.path.join(opt.output_dir, f"{image_name}_detections.json")
        with open(json_path, "w", encoding="ascii") as handle:
            json.dump(detections, handle, ensure_ascii=True, indent=2)
        log_info(f"[save] detections -> {json_path}")

    if opt.save_vis:
        vis_path = os.path.join(opt.output_dir, f"{image_name}_vis.jpg")
        visualize_detections(
            opt.original_image,
            detections,
            class_names=class_names,
            save_path=vis_path,
            keypoint_edges=getattr(opt, "keypoint_edges", []),
        )
        log_info(f"[save] visualization -> {vis_path}")


def main():
    default_config = os.path.join(_PROJECT_ROOT, "configs", "onnx_infer.yaml")
    parser = argparse.ArgumentParser(description="ONNX Runtime 单图推理")
    parser.add_argument("image", help="待推理的jpg图片路径")
    parser.add_argument("--config", default=default_config, help="YAML配置文件路径")
    args = parser.parse_args()

    cfg_dict, cfg_path = load_yaml_config(args.config)
    cfg_dict = cfg_dict or {}
    config_dir = os.path.dirname(cfg_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_dict["image"] = args.image

    cfg = SimpleNamespace(**cfg_dict)

    onnx_model_path = getattr(cfg, "onnx_model", "")
    if not onnx_model_path:
        log_error("Config缺少 'onnx_model' 字段")
        return
    if not os.path.isabs(onnx_model_path):
        onnx_model_path = resolve_relative_path(
            onnx_model_path,
            [config_dir, script_dir, _PROJECT_ROOT],
            must_exist=True,
        )
    if not os.path.exists(onnx_model_path):
        log_error(f"ONNX model not found: {onnx_model_path}")
        return
    cfg.onnx_model = onnx_model_path

    image_path = cfg.image
    if not os.path.isabs(image_path):
        image_path = resolve_relative_path(
            image_path,
            [config_dir, script_dir, _PROJECT_ROOT],
            must_exist=True,
        )
    if not os.path.exists(image_path):
        log_error(f"Image path not found: {image_path}")
        return

    image_paths = []
    if os.path.isdir(image_path):
        for entry in sorted(os.listdir(image_path)):
            entry_path = os.path.join(image_path, entry)
            if not os.path.isfile(entry_path):
                continue
            ext = os.path.splitext(entry)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_paths.append(entry_path)
        if not image_paths:
            log_error(f"No image files found in directory: {image_path}")
            return
    else:
        image_paths = [image_path]
    cfg.image = image_paths[0]

    output_dir = getattr(cfg, "output_dir", "result/onnx_infer")
    if not os.path.isabs(output_dir):
        output_dir = resolve_relative_path(
            output_dir,
            [config_dir, script_dir, _PROJECT_ROOT],
            must_exist=False,
        )
    cfg.output_dir = output_dir

    class_names_path = getattr(cfg, "class_names", "")
    if class_names_path and not os.path.isabs(class_names_path):
        class_names_path = resolve_relative_path(
            class_names_path,
            [config_dir, script_dir, _PROJECT_ROOT],
            must_exist=True,
        )
    cfg.class_names = class_names_path

    providers = getattr(cfg, "providers", [])
    if isinstance(providers, str):
        providers = [p.strip() for p in providers.split(",") if p.strip()]
    elif providers is None:
        providers = []
    cfg.providers = providers

    cfg.task = getattr(cfg, "task", "auto")
    cfg.vis_thresh = float(getattr(cfg, "vis_thresh", 0.3))
    cfg.cross_class_nms_thresh = float(getattr(cfg, "cross_class_nms_thresh", -1.0))
    cfg.input_rgb = bool(getattr(cfg, "input_rgb", False))
    cfg.save_vis = bool(getattr(cfg, "save_vis", False))
    cfg.save_json = bool(getattr(cfg, "save_json", False))
    cfg.print_results = bool(getattr(cfg, "print_results", False))
    cfg.num_classes = int(getattr(cfg, "num_classes", -1))
    cfg.num_joints = int(getattr(cfg, "num_joints", -1))
    cfg.topk = int(getattr(cfg, "topk", 100))
    cfg.max_dets = int(getattr(cfg, "max_dets", 50))
    cfg.input_h = int(getattr(cfg, "input_h", -1))
    cfg.input_w = int(getattr(cfg, "input_w", -1))

    mean_arr = _parse_triplet(getattr(cfg, "mean", [0.0, 0.0, 0.0]), (0.0, 0.0, 0.0))
    std_arr = _parse_triplet(getattr(cfg, "std", [1.0, 1.0, 1.0]), (1.0, 1.0, 1.0))

    class_names = None
    if cfg.class_names:
        base_names = _load_class_names(cfg.class_names)
        target_classes = cfg.num_classes if int(getattr(cfg, "num_classes", -1)) > 0 else None
        class_names = _normalize_class_names(base_names, target_classes)

    session_kwargs = {"providers": cfg.providers} if cfg.providers else {}
    ort_session = ort.InferenceSession(cfg.onnx_model, **session_kwargs)

    input_meta = ort_session.get_inputs()[0]
    input_h, input_w = _determine_input_size(input_meta.shape, cfg.input_h, cfg.input_w)

    _ensure_dir(cfg.output_dir)
    total_images = len(image_paths)
    output_names = [meta.name or f"head_{idx}" for idx, meta in enumerate(ort_session.get_outputs())]
    log_info(f"[onnx] outputs: {output_names}", detail=True)
    log_info(f"[init] collected {total_images} image(s) for inference.")

    for img_idx, img_path in enumerate(image_paths, start=1):
        cfg.image = img_path
        image_basename = os.path.splitext(os.path.basename(img_path))[0]
        log_info(f"[step] preprocessing image ({img_idx}/{total_images}): {img_path}")
        (
            input_tensor,
            original_img,
            original_size,
            center,
            scale,
        ) = preprocess_image(
            img_path,
            input_h,
            input_w,
            mean_arr,
            std_arr,
            use_rgb=cfg.input_rgb,
        )

        log_info(f"[step] running onnx inference ({img_idx}/{total_images})...")
        ort_inputs = {input_meta.name: input_tensor}
        ort_outputs = ort_session.run(None, ort_inputs)
        log_info(f"[onnx] output shapes: {[np.shape(out) for out in ort_outputs]}", detail=True)

        opt = _create_opt_namespace(
            cfg,
            ort_outputs,
            output_names,
            original_img,
            input_h,
            input_w,
            class_names,
        )

        log_info(f"[step] postprocessing detections ({img_idx}/{total_images})...")
        detections = postprocess_detections(ort_outputs, opt, center, scale)
        detections = detections[: opt.max_dets]
        log_info(f"[result] {image_basename}: detections kept -> {len(detections)}")

        if opt.print_results:
            print(f"[image] {image_basename} ({img_idx}/{total_images})")
            for det_idx, det in enumerate(detections):
                cls_id = int(det.get("class_id", -1))
                score = float(det.get("confidence", 0.0))
                bbox = det.get("bbox", [0, 0, 0, 0])
                cls_name = (
                    class_names[cls_id] if class_names and 0 <= cls_id < len(class_names) else f"class_{cls_id}"
                )
                print(
                    "  #{:02d}: cls={} ({}) score={:.3f} bbox=[{:.1f}, {:.1f}, {:.1f}, {:.1f}]".format(
                        det_idx, cls_id, cls_name, score, bbox[0], bbox[1], bbox[2], bbox[3]
                    )
                )

        _save_results(opt, image_basename, detections, class_names)

    log_info("[done] inference finished.")


if __name__ == "__main__":
    main()
