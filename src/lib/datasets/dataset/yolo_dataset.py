from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import json
import cv2
import os
import glob
from pathlib import Path
from collections import defaultdict


def _parse_flip_pairs(pairs_str, num_joints):
    if not pairs_str:
        return []
    pairs = []
    for chunk in pairs_str.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            left, right = chunk.split("-")
            i, j = int(left), int(right)
        except ValueError:
            raise ValueError("Invalid flip pair '{}'. Use the form index-index".format(chunk))
        if not (0 <= i < num_joints and 0 <= j < num_joints):
            raise ValueError("Flip pair '{}' indexes must be within [0, {})".format(chunk, num_joints))
        pairs.append([i, j])
    return pairs


def _infer_keypoint_config(dataset_dir):
    """Infer number of keypoints and components per keypoint from labels."""

    label_root = Path(dataset_dir) / "labels"
    if not label_root.exists():
        return 0, 0

    for split in ["train", "val", "test"]:
        split_dir = label_root / split
        if not split_dir.exists():
            continue
        for label_path in sorted(split_dir.glob("*.txt")):
            try:
                with label_path.open("r") as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if not parts or len(parts) <= 5:
                            continue
                        remainder = len(parts) - 5
                        if remainder <= 0:
                            continue
                        if remainder % 3 == 0:
                            return remainder // 3, 3
                        if remainder % 2 == 0:
                            return remainder // 2, 2
                        # Otherwise keep searching other lines/files.
            except OSError:
                continue

    return 0, 0


import torch.utils.data as data

_FORCED_NUM_CLASSES = 2
_DEFAULT_YOLO_CLASSES = ["class_{}".format(i) for i in range(_FORCED_NUM_CLASSES)]


_DEFAULT_COCO_CLASSES = [
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


def _resolve_dataset_dir(opt):
    data_dir = getattr(opt, "data_dir", None)
    if data_dir:
        base = Path(data_dir)
    else:
        base = Path(__file__).resolve().parents[4] / "data"
    return base / "yolo_dataset"


def _load_class_names(dataset_dir):
    class_file = dataset_dir / "classes.txt"
    if class_file.exists():
        names = [line.strip() for line in class_file.read_text().splitlines() if line.strip()]
        if names:
            names = names[:_FORCED_NUM_CLASSES]
            while len(names) < _FORCED_NUM_CLASSES:
                names.append("class_{}".format(len(names)))
            return names

    inferred_classes = _infer_num_classes(dataset_dir)
    if inferred_classes > 0:
        names = ["class_{}".format(i) for i in range(min(inferred_classes, _FORCED_NUM_CLASSES))]
        while len(names) < _FORCED_NUM_CLASSES:
            names.append("class_{}".format(len(names)))
        return names

    return list(_DEFAULT_YOLO_CLASSES)


def _infer_num_classes(dataset_dir):
    label_root = Path(dataset_dir) / "labels"
    if not label_root.exists():
        return 0
    max_class = -1
    for split in ["train", "val", "test"]:
        split_dir = label_root / split
        if not split_dir.exists():
            continue
        for label_path in sorted(split_dir.glob("*.txt")):
            try:
                with label_path.open("r") as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            cls_id = int(float(parts[0]))
                        except ValueError:
                            continue
                        if cls_id > max_class:
                            max_class = cls_id
                if max_class >= 1:
                    return min(max_class + 1, _FORCED_NUM_CLASSES)
            except OSError:
                continue
    return max_class + 1 if max_class >= 0 else 0


def _parse_mean_std_override(raw_value, flag_name):
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        tokens = [tok for tok in raw_value.replace(",", " ").split() if tok]
        if not tokens:
            return None
        try:
            values = [float(tok) for tok in tokens]
        except ValueError as exc:
            raise ValueError("{} expects three float values but got {!r}".format(flag_name, raw_value)) from exc
    else:
        try:
            arr = np.asarray(raw_value, dtype=np.float32).reshape(-1)
        except Exception as exc:
            raise ValueError("{} expects an iterable with three numeric values but got {!r}".format(flag_name, raw_value)) from exc
        values = [float(v) for v in arr.tolist()]
    if len(values) == 0:
        return None
    if len(values) != 3:
        raise ValueError("{} expects exactly three values, got {}".format(flag_name, len(values)))
    return values


class YOLODataset(data.Dataset):
    num_classes = len(_DEFAULT_COCO_CLASSES)  # Default to COCO classes
    default_resolution = [512, 512]
    mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)

    @classmethod
    def get_dataset_spec(cls, opt):
        dataset_dir = _resolve_dataset_dir(opt)
        class_names = _load_class_names(dataset_dir)
        cls.num_classes = len(class_names)

        requested_kpts = getattr(opt, "yolo_num_kpts", -1)
        inferred_joints, inferred_components = _infer_keypoint_config(dataset_dir)
        if requested_kpts >= 0:
            num_joints = requested_kpts
            keypoint_components = inferred_components if inferred_components else 2
        else:
            num_joints = inferred_joints
            keypoint_components = inferred_components

        if num_joints <= 0:
            keypoint_components = 0

        use_rgb = bool(getattr(opt, "input_rgb", False))
        try:
            override_mean = _parse_mean_std_override(getattr(opt, "yolo_mean", None), "--yolo_mean")
        except ValueError as exc:
            raise ValueError("Failed to parse --yolo_mean: {}".format(exc))
        try:
            override_std = _parse_mean_std_override(getattr(opt, "yolo_std", None), "--yolo_std")
        except ValueError as exc:
            raise ValueError("Failed to parse --yolo_std: {}".format(exc))

        mean_values = override_mean if override_mean is not None else cls.mean.reshape(-1).tolist()
        std_values = override_std if override_std is not None else cls.std.reshape(-1).tolist()
        if use_rgb:
            if override_mean is None:
                mean_values = mean_values[::-1]
            if override_std is None:
                std_values = std_values[::-1]
        mean_array = np.array(mean_values, dtype=np.float32).reshape(1, 1, 3)
        std_array = np.array(std_values, dtype=np.float32).reshape(1, 1, 3)

        spec = {
            "default_resolution": cls.default_resolution,
            "mean": mean_array,
            "std": std_array,
            "num_classes": cls.num_classes,
            "class_names": class_names,
            "dataset_dir": str(dataset_dir),
            "num_joints": max(0, num_joints),
        }

        if num_joints > 0:
            try:
                spec["flip_idx"] = _parse_flip_pairs(getattr(opt, "yolo_flip_pairs", ""), num_joints)
            except ValueError as exc:
                raise ValueError("Failed to parse --yolo_flip_pairs: {}".format(exc))
            spec["keypoint_edges"] = []
            spec["keypoint_components"] = max(2, keypoint_components or 2)
        else:
            spec["keypoint_components"] = 0
        return spec

    def __init__(self, opt, split):
        super(YOLODataset, self).__init__()
        spec = self.get_dataset_spec(opt)
        self.class_names = spec.get("class_names", list(_DEFAULT_COCO_CLASSES))
        self.num_classes = spec.get("num_classes", len(self.class_names))
        self.class_name = ["__background__"] + self.class_names
        self.mean = np.array(spec.get("mean", self.mean), dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(spec.get("std", self.std), dtype=np.float32).reshape(1, 1, 3)

        dataset_dir = spec.get("dataset_dir")
        self.data_dir = str(dataset_dir) if dataset_dir is not None else os.path.join(opt.data_dir, "yolo_dataset")
        self.img_dir = os.path.join(self.data_dir, "images", split)
        self.label_dir = os.path.join(self.data_dir, "labels", split)
        self.max_objs = 128
        self.num_joints = spec.get("num_joints", 0)
        self.keypoint_components = spec.get("keypoint_components", 0)
        self.flip_idx = spec.get("flip_idx", [])
        self.keypoint_edges = spec.get("keypoint_edges", [])
        self._has_keypoints = self.num_joints > 0

        # Create class mapping (COCO-style, 1-indexed category ids)
        self._valid_ids = np.arange(1, self.num_classes + 1, dtype=np.int32)
        self.cat_ids = {int(cat_id): idx for idx, cat_id in enumerate(self._valid_ids)}

        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array(
            [
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938],
            ],
            dtype=np.float32,
        )

        self.split = split
        self.opt = opt

        # Load image paths and annotations
        self.image_info = []
        self.annotations = []
        self._load_data()

        # Create images list as expected by CTDet (image IDs)
        self.images = list(range(len(self.image_info)))

        # For compatibility with COCO-based code
        self.coco = self

        print("==> initializing YOLO {} data.".format(split))
        print("Loaded {} {} samples".format(split, len(self.images)))

    def _load_data(self):
        """Load YOLO format data"""
        # Get all image files
        img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        img_files = []
        for ext in img_extensions:
            img_files.extend(glob.glob(os.path.join(self.img_dir, ext)))
            img_files.extend(glob.glob(os.path.join(self.img_dir, ext.upper())))

        img_files = sorted(img_files)

        for img_path in img_files:
            # Get corresponding label file
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.label_dir, img_name + ".txt")

            # Load image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                continue

            height, width = img.shape[:2]

            # Load annotations
            annotations = []
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    expected_len = 5 + (self.num_joints * self.keypoint_components if self._has_keypoints else 0)

                    if float(parts[0]) == -1:
                        continue
                    if self._has_keypoints and len(parts) != expected_len:
                        raise ValueError(
                            "Label '{}' expected {} values (cls + bbox + {} keypoints) but found {}".format(
                                label_path, expected_len, self.num_joints, len(parts)
                            )
                        )
                    if not self._has_keypoints and len(parts) != 5:
                        continue

                    class_id = int(float(parts[0]))
                    if class_id < 0 or class_id >= self.num_classes:
                        continue
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    # Convert YOLO format to absolute coordinates
                    x_center_abs = x_center * width
                    y_center_abs = y_center * height
                    w_abs = w * width
                    h_abs = h * height

                    # Convert to x1, y1, x2, y2
                    x1 = x_center_abs - w_abs / 2
                    y1 = y_center_abs - h_abs / 2
                    x2 = x_center_abs + w_abs / 2
                    y2 = y_center_abs + h_abs / 2

                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))

                    if x2 > x1 and y2 > y1:  # Valid bounding box
                        ann = {
                            "bbox": [x1, y1, x2 - x1, y2 - y1],  # x, y, width, height
                            "category_id": class_id + 1,  # 1-indexed for compatibility
                            "area": (x2 - x1) * (y2 - y1),
                            "iscrowd": 0,
                        }

                        if self._has_keypoints:
                            kp_values = [float(v) for v in parts[5:]]
                            keypoints = []
                            visible_count = 0
                            for j in range(self.num_joints):
                                offset = j * self.keypoint_components
                                kp_x = kp_values[offset]
                                kp_y = kp_values[offset + 1]
                                if self.keypoint_components == 3:
                                    kp_vis = kp_values[offset + 2]
                                else:
                                    kp_vis = 1.0

                                if 0.0 <= kp_x <= 1.0 and 0.0 <= kp_y <= 1.0:
                                    kp_x *= width
                                    kp_y *= height

                                if kp_vis > 0:
                                    visible_count += 1

                                keypoints.extend([kp_x, kp_y, kp_vis])

                            ann["keypoints"] = keypoints
                            ann["num_keypoints"] = visible_count

                        annotations.append(ann)

            self.image_info.append(
                {
                    "file_name": os.path.basename(img_path),
                    "height": height,
                    "width": width,
                    "id": len(self.image_info),
                    "path": img_path,
                }
            )
            self.annotations.append(annotations)

    def __len__(self):
        return len(self.images)

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        """Convert evaluation format for compatibility"""
        detections = [[[] for __ in range(len(self.images))] for _ in range(self.num_classes + 1)]

        for i in range(len(self.images)):
            img_id = i
            for j in range(1, self.num_classes + 1):
                if img_id in all_bboxes and j in all_bboxes[img_id]:
                    if isinstance(all_bboxes[img_id][j], np.ndarray):
                        detections[j][i] = all_bboxes[img_id][j].tolist()
                    else:
                        detections[j][i] = all_bboxes[img_id][j]
                else:
                    detections[j][i] = []
        return detections

    def save_results(self, results, save_dir):
        """Save results to JSON file"""
        json.dump(self.convert_eval_format(results), open("{}/results.json".format(save_dir), "w"))

    def run_eval(self, results, save_dir):
        """Run evaluation"""
        self.save_results(results, save_dir)
        print("Results saved to {}/results.json".format(save_dir))

        # Simple evaluation metrics
        total_detections = 0
        for img_results in results.values():
            for class_results in img_results.values():
                if isinstance(class_results, (list, np.ndarray)):
                    total_detections += len(class_results)

        print("Total detections: {}".format(total_detections))
        print("Average detections per image: {:.2f}".format(total_detections / len(self.images)))

    def loadImgs(self, ids):
        """Load image info for compatibility with COCO API"""
        result = []
        for img_id in ids:
            if isinstance(img_id, dict):
                result.append(img_id)
                continue
            try:
                idx = int(img_id)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(self.image_info):
                result.append(self.image_info[idx])
        return result

    def getAnnIds(self, imgIds):
        """Get annotation IDs for compatibility with COCO API"""
        if isinstance(imgIds, int):
            img_ids = [int(imgIds)]
        else:
            img_ids = []
            for item in imgIds or []:
                try:
                    img_ids.append(int(item))
                except (TypeError, ValueError):
                    continue
        if not img_ids:
            return []
        img_id = img_ids[0]
        if 0 <= img_id < len(self.annotations):
            self._current_img_id = img_id  # Store current image ID for loadAnns
            return list(range(len(self.annotations[img_id])))
        return []

    def loadAnns(self, ids):
        """Load annotations for compatibility with COCO API"""
        if not hasattr(self, "_current_img_id"):
            return []
        img_id = self._current_img_id
        if not (0 <= img_id < len(self.annotations)):
            return []
        anns = self.annotations[img_id]
        if not ids:
            return anns
        return [anns[i] for i in ids if 0 <= i < len(anns)]
