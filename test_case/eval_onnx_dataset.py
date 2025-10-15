#!/usr/bin/env python3
"""
ONNX dataset evaluation
-----------------------
Batch-evaluate a converted CenterNet ONNX model over specified dataset splits,
computing detection recall/precision/IoU together with keypoint mean L2 errors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np

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

_CWD = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CWD, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from test_case import infer_onnx_runtime as infer_mod  # noqa: E402

preprocess_image = infer_mod.preprocess_image
postprocess_detections = infer_mod.postprocess_detections
log_info = infer_mod.log_info
log_warn = infer_mod.log_warn
log_error = infer_mod.log_error
_apply_cross_class_nms = getattr(infer_mod, "_apply_cross_class_nms", None)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIS_CATEGORIES = (0, 1)


def load_yolo_ground_truth(label_path, original_size, num_joints, num_classes=None):
    gt = []
    width, height = original_size
    if not os.path.exists(label_path):
        return gt
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except OSError as exc:
        log_error(f"[ground_truth] failed to read label file: {label_path} | error: {exc}")
        return gt

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cls_id = max(0, cls_id)
        if num_classes is not None and num_classes > 0:
            cls_id = min(cls_id, num_classes - 1)
        xc, yc, w, h = map(float, parts[1:5])
        xc *= width
        yc *= height
        w *= width
        h *= height
        x1 = xc - w / 2.0
        y1 = yc - h / 2.0
        x2 = xc + w / 2.0
        y2 = yc + h / 2.0
        entry = {
            "class_id": cls_id,
            "bbox": [x1, y1, x2, y2],
            "confidence": 1.0,
            "keypoints": [],
        }

        remaining = parts[5:]
        if num_joints > 0:
            kps = []
            vis = []
            if len(remaining) >= num_joints * 3:
                for j in range(num_joints):
                    base = 3 * j
                    kx = float(remaining[base])
                    ky = float(remaining[base + 1])
                    kv = float(remaining[base + 2])
                    if 0.0 <= kx <= 1.0:
                        kx *= width
                    if 0.0 <= ky <= 1.0:
                        ky *= height
                    kps.append((kx, ky))
                    vis.append(kv)
            elif len(remaining) >= num_joints * 2:
                for j in range(num_joints):
                    kx = float(remaining[2 * j])
                    ky = float(remaining[2 * j + 1])
                    if 0.0 <= kx <= 1.0:
                        kx *= width
                    if 0.0 <= ky <= 1.0:
                        ky *= height
                    kps.append((kx, ky))
                    vis.append(1.0)
            if kps:
                entry["keypoints"] = kps
            if vis:
                entry["keypoint_visibility"] = vis
        gt.append(entry)
    return gt


def evaluate_detection_metrics(detections, gt_entries, num_joints=0, iou_thresh=0.5, verbose=True):
    if not gt_entries:
        if verbose:
            log_info("[eval] ground-truth annotations not provided, skip evaluation", detail=True)
        return None

    total_gt = len(gt_entries)
    total_det = len(detections)
    if total_det == 0:
        if verbose:
            log_info(f"[eval] no detections available to compare with {total_gt} ground-truth entries")
        return {
            "matches": 0,
            "total_gt": total_gt,
            "total_det": total_det,
            "recall": 0.0,
            "precision": 0.0,
            "mean_iou": 0.0,
            "iou_sum": 0.0,
            "keypoint_mean_l2": 0.0,
            "keypoint_total_count": 0,
            "keypoint_total_l2": 0.0,
            "per_joint_sums": [],
            "per_joint_counts": [],
        }

    if iou_thresh is None:
        iou_thresh = 0.0
    iou_thresh = max(0.0, float(iou_thresh))

    detections_sorted = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
    used_gt = set()
    matches = []
    unmatched_det = []

    for det in detections_sorted:
        best_iou = -1.0
        best_idx = -1
        det_bbox = det.get("bbox", [0.0, 0.0, 0.0, 0.0])
        det_cls = int(det.get("class_id", -1))
        for idx, gt in enumerate(gt_entries):
            if idx in used_gt:
                continue
            gt_cls = int(gt.get("class_id", -1))
            if gt_cls != det_cls:
                continue
            iou = infer_mod._bbox_iou(det_bbox, gt.get("bbox", [0.0, 0.0, 0.0, 0.0]))
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_thresh:
            used_gt.add(best_idx)
            matches.append((det, gt_entries[best_idx], best_iou))
        else:
            unmatched_det.append(det)

    match_count = len(matches)
    recall = match_count / total_gt if total_gt > 0 else 0.0
    precision = match_count / total_det if total_det > 0 else 0.0
    iou_values = [m[2] for m in matches]
    sum_iou = float(np.sum(iou_values)) if iou_values else 0.0
    mean_iou = sum_iou / match_count if match_count > 0 else 0.0

    if verbose:
        log_info(
            (
                "[eval] IoU_thresh={:.2f} | matches={} | total_gt={} | total_det={} | "
                "recall={:.3f} | precision={:.3f} | mean_IoU={:.3f}"
            ).format(iou_thresh, match_count, total_gt, total_det, recall, precision, mean_iou)
        )

    unmatched_gt = total_gt - match_count
    if verbose and (unmatched_gt or unmatched_det):
        log_info(
            f"[eval] unmatched counts -> gt:{unmatched_gt} det:{len(unmatched_det)}",
            detail=True,
        )

    import math

    per_joint_sums = None
    per_joint_counts = None
    total_kp_l2 = 0.0
    total_kp_count = 0
    vis_total_l2 = {cat: 0.0 for cat in VIS_CATEGORIES}
    vis_total_count = {cat: 0 for cat in VIS_CATEGORIES}
    vis_per_joint_sums = {cat: None for cat in VIS_CATEGORIES}
    vis_per_joint_counts = {cat: None for cat in VIS_CATEGORIES}
    vis_pred_correct = {cat: 0 for cat in VIS_CATEGORIES}
    vis_pred_total = {cat: 0 for cat in VIS_CATEGORIES}

    target_joint_count = int(num_joints) if int(num_joints) > 0 else 0

    for det, gt, _ in matches:
        det_kps = det.get("keypoints") or []
        gt_kps = gt.get("keypoints") or []
        gt_vis_list = gt.get("keypoint_visibility") or []
        det_vis_list = det.get("keypoint_visibility") or []
        if not det_kps or not gt_kps:
            continue
        usable = min(len(det_kps), len(gt_kps))
        if target_joint_count > 0:
            usable = min(usable, target_joint_count)
        if usable <= 0:
            continue
        if per_joint_sums is None:
            size = target_joint_count if target_joint_count > 0 else usable
            per_joint_sums = [0.0] * size
            per_joint_counts = [0] * size
        if usable > len(per_joint_sums):
            extra = usable - len(per_joint_sums)
            per_joint_sums.extend([0.0] * extra)
            per_joint_counts.extend([0] * extra)
        for idx in range(usable):
            gt_vis = None
            if idx < len(gt_vis_list):
                try:
                    gt_vis = int(round(float(gt_vis_list[idx])))
                except (TypeError, ValueError):
                    gt_vis = None
            if gt_vis not in VIS_CATEGORIES:
                continue
            dx = float(det_kps[idx][0]) - float(gt_kps[idx][0])
            dy = float(det_kps[idx][1]) - float(gt_kps[idx][1])
            l2 = math.hypot(dx, dy)
            per_joint_sums[idx] += l2
            per_joint_counts[idx] += 1
            total_kp_l2 += l2
            total_kp_count += 1

            if vis_per_joint_sums[gt_vis] is None:
                size = target_joint_count if target_joint_count > 0 else usable
                vis_per_joint_sums[gt_vis] = [0.0] * size
                vis_per_joint_counts[gt_vis] = [0] * size
            if idx >= len(vis_per_joint_sums[gt_vis]):
                extra = idx + 1 - len(vis_per_joint_sums[gt_vis])
                vis_per_joint_sums[gt_vis].extend([0.0] * extra)
                vis_per_joint_counts[gt_vis].extend([0] * extra)
            vis_per_joint_sums[gt_vis][idx] += l2
            vis_per_joint_counts[gt_vis][idx] += 1
            vis_total_l2[gt_vis] += l2
            vis_total_count[gt_vis] += 1

            pred_flag = None
            if idx < len(det_vis_list):
                try:
                    pred_score = float(det_vis_list[idx])
                    pred_flag = 1 if pred_score >= 0.5 else 0
                except (TypeError, ValueError):
                    pred_flag = None
            if pred_flag is not None:
                vis_pred_total[gt_vis] += 1
                if pred_flag == gt_vis:
                    vis_pred_correct[gt_vis] += 1

    if total_kp_count > 0:
        mean_l2 = total_kp_l2 / total_kp_count
        joint_means = []
        for idx, count in enumerate(per_joint_counts or []):
            if count > 0:
                joint_means.append((idx, per_joint_sums[idx] / count))
        joint_str = ", ".join(f"j{idx}:{val:.2f}" for idx, val in joint_means)
        if verbose:
            log_info(f"[eval] keypoint mean L2={mean_l2:.3f}px" + (f" | per-joint: {joint_str}" if joint_str else ""))
    else:
        mean_l2 = 0.0
        per_joint_sums = per_joint_sums or []
        per_joint_counts = per_joint_counts or []
        if verbose:
            log_info("[eval] keypoint metrics unavailable (no matched keypoints)", detail=True)

    return {
        "matches": match_count,
        "total_gt": total_gt,
        "total_det": total_det,
        "recall": recall,
        "precision": precision,
        "mean_iou": mean_iou,
        "iou_sum": sum_iou,
        "keypoint_mean_l2": mean_l2,
        "keypoint_total_count": total_kp_count,
        "keypoint_total_l2": total_kp_l2,
        "per_joint_sums": per_joint_sums if per_joint_sums is not None else [],
        "per_joint_counts": per_joint_counts if per_joint_counts is not None else [],
        "vis_total_l2": vis_total_l2,
        "vis_total_count": vis_total_count,
        "vis_per_joint_sums": {
            cat: (vis_per_joint_sums[cat] if vis_per_joint_sums[cat] is not None else []) for cat in VIS_CATEGORIES
        },
        "vis_per_joint_counts": {
            cat: (vis_per_joint_counts[cat] if vis_per_joint_counts[cat] is not None else []) for cat in VIS_CATEGORIES
        },
        "vis_pred_correct": vis_pred_correct,
        "vis_pred_total": vis_pred_total,
    }


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


def collect_images(image_dir):
    pairs = []
    if not os.path.isdir(image_dir):
        log_warn(f"[collect] image directory not found: {image_dir}")
        return pairs
    for entry in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(entry)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        pairs.append(os.path.join(image_dir, entry))
    return pairs


def build_opt_namespace(cfg):
    opt = SimpleNamespace()
    opt.task = cfg.task
    opt.num_classes = cfg.num_classes
    opt.num_joints = cfg.num_joints
    opt.K = cfg.topk
    opt.vis_thresh = cfg.vis_thresh
    opt.cross_class_nms_thresh = cfg.cross_class_nms_thresh
    opt.max_dets = cfg.max_dets
    opt.reg_offset = False
    opt.hm_hp = False
    opt.reg_hp_offset = False
    opt.cat_spec_wh = False
    opt.heads = {}
    return opt


def evaluate_single_image(
    img_path,
    label_path,
    session,
    input_name,
    cfg,
    mean_arr,
    std_arr,
):
    (
        input_tensor,
        original_img,
        original_size_hw,
        center,
        scale,
    ) = preprocess_image(
        img_path,
        cfg.input_h,
        cfg.input_w,
        mean_arr,
        std_arr,
        use_rgb=cfg.input_rgb,
    )

    ort_outputs = session.run(None, {input_name: input_tensor})
    output_names = [meta.name or f"head_{idx}" for idx, meta in enumerate(session.get_outputs())]

    opt = build_opt_namespace(cfg)
    heads = {}
    for name, output in zip(output_names, ort_outputs):
        channels = output.shape[1] if output.ndim >= 2 else output.shape[-1]
        heads[name] = int(channels)
    opt.heads = heads
    opt.reg_offset = "reg" in heads
    opt.hm_hp = "hm_hp" in heads
    opt.reg_hp_offset = "hp_offset" in heads

    detections = postprocess_detections(ort_outputs, opt, center, scale)
    if opt.max_dets and opt.max_dets > 0:
        detections = detections[: opt.max_dets]

    orig_h, orig_w = original_size_hw
    original_size_wh = (float(orig_w), float(orig_h))

    gt_entries = []
    if label_path and os.path.exists(label_path):
        gt_entries = load_yolo_ground_truth(
            label_path,
            original_size_wh,
            cfg.num_joints,
            cfg.num_classes,
        )
    else:
        log_warn(f"[eval] label not found for image: {img_path}")

    metrics = evaluate_detection_metrics(
        detections,
        gt_entries,
        cfg.num_joints,
        cfg.eval_iou_thresh,
        verbose=False,
    )

    return metrics


def aggregate_metrics(accum, metrics):
    if metrics is None:
        return
    accum["images"] += 1
    accum["total_gt"] += metrics["total_gt"]
    accum["total_det"] += metrics["total_det"]
    accum["matches"] += metrics["matches"]
    accum["iou_sum"] += metrics["iou_sum"]
    accum["kp_total_l2"] += metrics["keypoint_total_l2"]
    accum["kp_total_count"] += metrics["keypoint_total_count"]

    per_joint_sums = metrics.get("per_joint_sums") or []
    per_joint_counts = metrics.get("per_joint_counts") or []
    if per_joint_sums and len(accum["per_joint_sums"]) < len(per_joint_sums):
        extra = len(per_joint_sums) - len(accum["per_joint_sums"])
        accum["per_joint_sums"].extend([0.0] * extra)
        accum["per_joint_counts"].extend([0] * extra)
    for idx, val in enumerate(per_joint_sums):
        accum["per_joint_sums"][idx] += float(val)
    for idx, count in enumerate(per_joint_counts):
        accum["per_joint_counts"][idx] += int(count)

    if "vis_total_l2" not in accum:
        accum["vis_total_l2"] = {cat: 0.0 for cat in VIS_CATEGORIES}
        accum["vis_total_count"] = {cat: 0 for cat in VIS_CATEGORIES}
        accum["vis_per_joint_sums"] = {cat: [] for cat in VIS_CATEGORIES}
        accum["vis_per_joint_counts"] = {cat: [] for cat in VIS_CATEGORIES}
        accum["vis_pred_correct"] = {cat: 0 for cat in VIS_CATEGORIES}
        accum["vis_pred_total"] = {cat: 0 for cat in VIS_CATEGORIES}

    metric_vis_total_l2 = metrics.get("vis_total_l2", {})
    metric_vis_total_count = metrics.get("vis_total_count", {})
    metric_vis_per_joint_sums = metrics.get("vis_per_joint_sums", {})
    metric_vis_per_joint_counts = metrics.get("vis_per_joint_counts", {})
    metric_vis_pred_correct = metrics.get("vis_pred_correct", {})
    metric_vis_pred_total = metrics.get("vis_pred_total", {})

    for cat in VIS_CATEGORIES:
        accum["vis_total_l2"][cat] += float(metric_vis_total_l2.get(cat, 0.0))
        accum["vis_total_count"][cat] += int(metric_vis_total_count.get(cat, 0))
        sums = metric_vis_per_joint_sums.get(cat) or []
        counts = metric_vis_per_joint_counts.get(cat) or []
        if sums and len(accum["vis_per_joint_sums"][cat]) < len(sums):
            extra = len(sums) - len(accum["vis_per_joint_sums"][cat])
            accum["vis_per_joint_sums"][cat].extend([0.0] * extra)
            accum["vis_per_joint_counts"][cat].extend([0] * extra)
        for idx, val in enumerate(sums):
            accum["vis_per_joint_sums"][cat][idx] += float(val)
        for idx, count in enumerate(counts):
            accum["vis_per_joint_counts"][cat][idx] += int(count)
        accum.setdefault("vis_pred_correct", {cat: 0 for cat in VIS_CATEGORIES})
        accum.setdefault("vis_pred_total", {cat: 0 for cat in VIS_CATEGORIES})
        accum["vis_pred_correct"][cat] += int(metric_vis_pred_correct.get(cat, 0))
        accum["vis_pred_total"][cat] += int(metric_vis_pred_total.get(cat, 0))


def finalize_metrics(accum):
    matches = accum["matches"]
    total_gt = accum["total_gt"]
    total_det = accum["total_det"]
    precision = matches / total_det if total_det > 0 else 0.0
    recall = matches / total_gt if total_gt > 0 else 0.0
    mean_iou = accum["iou_sum"] / matches if matches > 0 else 0.0
    kp_mean = accum["kp_total_l2"] / accum["kp_total_count"] if accum["kp_total_count"] > 0 else 0.0

    per_joint_means = []
    for idx, count in enumerate(accum["per_joint_counts"]):
        if count > 0:
            per_joint_means.append((idx, accum["per_joint_sums"][idx] / count))

    vis_summary = {}
    if "vis_total_l2" in accum:
        for cat in VIS_CATEGORIES:
            cat_total = accum["vis_total_count"].get(cat, 0)
            cat_mean = accum["vis_total_l2"].get(cat, 0.0) / cat_total if cat_total > 0 else 0.0
            joint_means_cat = []
            sums = accum["vis_per_joint_sums"].get(cat, [])
            counts = accum["vis_per_joint_counts"].get(cat, [])
            for idx, count in enumerate(counts):
                if count > 0:
                    joint_means_cat.append((idx, sums[idx] / count))
            correct = 0
            pred_total = 0
            if "vis_pred_correct" in accum and "vis_pred_total" in accum:
                correct = accum["vis_pred_correct"].get(cat, 0)
                pred_total = accum["vis_pred_total"].get(cat, 0)
            acc = correct / pred_total if pred_total > 0 else None
            vis_summary[cat] = {
                "mean": cat_mean,
                "per_joint": joint_means_cat,
                "count": cat_total,
                "vis_acc": acc,
                "pred_total": pred_total,
            }

    return {
        "images": accum["images"],
        "total_gt": total_gt,
        "total_det": total_det,
        "matches": matches,
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "kp_mean_l2": kp_mean,
        "per_joint_means": per_joint_means,
        "vis_summary": vis_summary,
    }


def format_split_result(split, summary):
    per_joint_text = ", ".join(f"j{idx}:{val:.2f}" for idx, val in summary.get("per_joint_means", [])) or "N/A"
    vis_summary = summary.get("vis_summary", {})
    vis_segments = []
    for cat in VIS_CATEGORIES:
        cat_data = vis_summary.get(cat, {})
        count = cat_data.get("count", 0)
        if count <= 0:
            vis_segments.append(f"vis={cat}: N/A")
            continue
        cat_mean = cat_data.get("mean", 0.0)
        cat_joint = ", ".join(f"j{idx}:{val:.2f}" for idx, val in cat_data.get("per_joint", [])) or "N/A"
        acc = cat_data.get("vis_acc", None)
        acc_text = f"acc={acc:.3f}" if acc is not None else "acc=N/A"
        vis_segments.append(f"vis={cat}: {acc_text} mean={cat_mean:.3f}px ({cat_joint})")
    vis_text = " | ".join(vis_segments)
    log_info(
        (
            "[split:{split}] images={images} gt={gt} det={det} matches={matches} "
            "recall={recall:.3f} precision={precision:.3f} mean_IoU={miou:.3f} "
            "kp_mean_L2={kp:.3f}px per_joint=({pj}) {vis}"
        ).format(
            split=split,
            images=summary["images"],
            gt=summary["total_gt"],
            det=summary["total_det"],
            matches=summary["matches"],
            recall=summary["recall"],
            precision=summary["precision"],
            miou=summary["mean_iou"],
            kp=summary["kp_mean_l2"],
            pj=per_joint_text,
            vis=vis_text,
        )
    )


#  python test_case/eval_onnx_dataset.py --config test_case/onnx_infer.yaml   --dataset_root data/yolo_annotations/ --splits train val
def main():
    default_config = os.path.join(_PROJECT_ROOT, "configs", "onnx_infer.yaml")
    parser = argparse.ArgumentParser(description="Evaluate ONNX model metrics on dataset splits")
    parser.add_argument("--config", default=default_config, help="YAML配置文件路径")
    parser.add_argument(
        "--dataset_root",
        default=os.path.join(_PROJECT_ROOT, "data", "yolo_annotations"),
        help="数据集根目录，需包含 images/ 与 labels/",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="需要评估的子集名称（对应 images/<split> 与 labels/<split>）",
    )
    parser.add_argument("--image_subdir", default="images", help="图像目录子路径")
    parser.add_argument("--label_subdir", default="labels", help="标签目录子路径")
    parser.add_argument("--limit", type=int, default=-1, help="每个子集最多评估的样本数，-1 表示全部")
    args = parser.parse_args()

    cfg_dict, cfg_path = load_yaml_config(args.config)
    cfg_dict = cfg_dict or {}
    config_dir = os.path.dirname(cfg_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    cfg = SimpleNamespace(**cfg_dict)
    cfg.task = getattr(cfg, "task", "auto")
    cfg.vis_thresh = float(getattr(cfg, "vis_thresh", 0.3))
    cfg.cross_class_nms_thresh = float(getattr(cfg, "cross_class_nms_thresh", -1.0))
    cfg.input_rgb = bool(getattr(cfg, "input_rgb", False))
    cfg.num_classes = int(getattr(cfg, "num_classes", -1))
    cfg.num_joints = int(getattr(cfg, "num_joints", 0))
    cfg.topk = int(getattr(cfg, "topk", 100))
    cfg.max_dets = int(getattr(cfg, "max_dets", 50))
    cfg.input_h = int(getattr(cfg, "input_h", -1))
    cfg.input_w = int(getattr(cfg, "input_w", -1))
    cfg.eval_iou_thresh = float(getattr(cfg, "eval_iou_thresh", 0.5))

    if cfg.input_h <= 0 or cfg.input_w <= 0:
        raise ValueError("input_h/input_w must be specified in config for evaluation.")

    dataset_root = args.dataset_root
    if not os.path.isabs(dataset_root):
        dataset_root = os.path.join(_PROJECT_ROOT, dataset_root)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    onnx_model_path = getattr(cfg, "onnx_model", "")
    if not onnx_model_path:
        raise ValueError("Config缺少 'onnx_model' 字段")
    if not os.path.isabs(onnx_model_path):
        onnx_model_path = resolve_relative_path(
            onnx_model_path,
            [config_dir, script_dir, _PROJECT_ROOT],
            must_exist=True,
        )
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

    mean_arr = np.array(getattr(cfg, "mean", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(getattr(cfg, "std", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(1, 1, 3)

    providers = getattr(cfg, "providers", []) or []
    if isinstance(providers, str):
        providers = [p.strip() for p in providers.split(",") if p.strip()]
    session_kwargs = {"providers": providers} if providers else {}
    session = ort.InferenceSession(onnx_model_path, **session_kwargs)
    input_name = session.get_inputs()[0].name

    log_info(f"[init] evaluating model={onnx_model_path}")
    log_info(f"[init] dataset_root={dataset_root} splits={args.splits}")
    log_info(f"[init] IoU threshold={cfg.eval_iou_thresh:.2f}, keypoints={cfg.num_joints}")

    overall = {
        "images": 0,
        "total_gt": 0,
        "total_det": 0,
        "matches": 0,
        "iou_sum": 0.0,
        "kp_total_l2": 0.0,
        "kp_total_count": 0,
        "per_joint_sums": [],
        "per_joint_counts": [],
        "vis_total_l2": {cat: 0.0 for cat in VIS_CATEGORIES},
        "vis_total_count": {cat: 0 for cat in VIS_CATEGORIES},
        "vis_per_joint_sums": {cat: [] for cat in VIS_CATEGORIES},
        "vis_per_joint_counts": {cat: [] for cat in VIS_CATEGORIES},
        "vis_pred_correct": {cat: 0 for cat in VIS_CATEGORIES},
        "vis_pred_total": {cat: 0 for cat in VIS_CATEGORIES},
    }

    for split in args.splits:
        image_dir = os.path.join(dataset_root, args.image_subdir, split)
        label_dir = os.path.join(dataset_root, args.label_subdir, split)
        image_paths = collect_images(image_dir)
        if not image_paths:
            log_warn(f"[split:{split}] no images found in {image_dir}, skip.")
            continue
        if args.limit > 0:
            image_paths = image_paths[: args.limit]

        log_info(f"[split:{split}] evaluating {len(image_paths)} images...")
        accum = {
            "images": 0,
            "total_gt": 0,
            "total_det": 0,
            "matches": 0,
            "iou_sum": 0.0,
            "kp_total_l2": 0.0,
            "kp_total_count": 0,
            "per_joint_sums": [],
            "per_joint_counts": [],
            "vis_total_l2": {cat: 0.0 for cat in VIS_CATEGORIES},
            "vis_total_count": {cat: 0 for cat in VIS_CATEGORIES},
            "vis_per_joint_sums": {cat: [] for cat in VIS_CATEGORIES},
            "vis_per_joint_counts": {cat: [] for cat in VIS_CATEGORIES},
            "vis_pred_correct": {cat: 0 for cat in VIS_CATEGORIES},
            "vis_pred_total": {cat: 0 for cat in VIS_CATEGORIES},
        }

        for idx, img_path in enumerate(image_paths, start=1):
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, f"{base}.txt")
            metrics = evaluate_single_image(
                img_path,
                label_path,
                session,
                input_name,
                cfg,
                mean_arr,
                std_arr,
            )
            if metrics is None:
                continue
            aggregate_metrics(accum, metrics)

        summary = finalize_metrics(accum)
        format_split_result(split, summary)
        overall["images"] += accum["images"]
        overall["total_gt"] += accum["total_gt"]
        overall["total_det"] += accum["total_det"]
        overall["matches"] += accum["matches"]
        overall["iou_sum"] += accum["iou_sum"]
        overall["kp_total_l2"] += accum["kp_total_l2"]
        overall["kp_total_count"] += accum["kp_total_count"]
        if len(overall["per_joint_sums"]) < len(accum["per_joint_sums"]):
            extra = len(accum["per_joint_sums"]) - len(overall["per_joint_sums"])
            overall["per_joint_sums"].extend([0.0] * extra)
            overall["per_joint_counts"].extend([0] * extra)
        for idx, val in enumerate(accum["per_joint_sums"]):
            overall["per_joint_sums"][idx] += float(val)
        for idx, count in enumerate(accum["per_joint_counts"]):
            overall["per_joint_counts"][idx] += int(count)
        for cat in VIS_CATEGORIES:
            overall["vis_total_l2"][cat] += accum["vis_total_l2"][cat]
            overall["vis_total_count"][cat] += accum["vis_total_count"][cat]
            if len(overall["vis_per_joint_sums"][cat]) < len(accum["vis_per_joint_sums"][cat]):
                extra = len(accum["vis_per_joint_sums"][cat]) - len(overall["vis_per_joint_sums"][cat])
                overall["vis_per_joint_sums"][cat].extend([0.0] * extra)
                overall["vis_per_joint_counts"][cat].extend([0] * extra)
            for idx, val in enumerate(accum["vis_per_joint_sums"][cat]):
                overall["vis_per_joint_sums"][cat][idx] += float(val)
            for idx, cnt in enumerate(accum["vis_per_joint_counts"][cat]):
                overall["vis_per_joint_counts"][cat][idx] += int(cnt)
            overall["vis_pred_correct"][cat] += accum["vis_pred_correct"][cat]
            overall["vis_pred_total"][cat] += accum["vis_pred_total"][cat]

    overall_summary = finalize_metrics(overall)
    format_split_result("overall", overall_summary)


if __name__ == "__main__":
    main()
