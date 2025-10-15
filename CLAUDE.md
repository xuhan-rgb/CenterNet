# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CenterNet implementation for object detection, 3D detection, and multi-pose estimation using center point detection. The project has been extended with YOLO format dataset support and custom keypoint detection capabilities.

**Key modifications from original CenterNet:**
- Added YOLO format dataset support (`yolo_dataset.py`)
- Supports custom number of keypoints per object
- Added ONNX export functionality during training
- Custom preprocessing with configurable mean/std normalization
- RGB input support (with BGR-to-RGB conversion)

## Python Environment

**CRITICAL:** All Python commands must be executed in the `py310` conda environment:
```bash
source ~/anaconda3/bin/activate && conda activate py310 && <your-python-command>
```

For Docker testing:
```bash
docker exec -it det_docker /bin/bash
# Then run commands inside container
```

## Project Configuration

**Input Parameters (from global CLAUDE.md):**
- Input resolution: 640x320 (width x height)
- Number of classes: 2
- These are crop detection parameters for the specific use case

## Training

### Train Keypoint Detection Model
```bash
bash train_keypoints.sh
```

This script trains a multi-pose model with:
- Dataset: `yolo_dataset` (YOLO format)
- Architecture: ResNet-18 by default
- Input size: 320h x 640w
- 4 keypoints per object
- Mixed precision training (AMP)
- RGB input mode
- Custom normalization: mean=[0,0,0], std=[1,1,1]

Training parameters are configured in `src/main.py` via command-line arguments.

### Key Training Options (from `src/lib/opts.py`)

**Dataset options:**
- `--dataset yolo_dataset`: Use YOLO format dataset
- `--yolo_num_kpts <N>`: Number of keypoints per object (-1 to auto-detect)
- `--yolo_flip_pairs "0-1,2-3"`: Define horizontal flip swaps for keypoints
- `--yolo_mean "0 0 0"`: Override normalization mean (BGR order)
- `--yolo_std "1 1 1"`: Override normalization std (BGR order)
- `--input_rgb`: Convert from BGR to RGB before normalization

**Keypoint training options:**
- `--learn_invisible_kpts`: Learn from invisible keypoints instead of masking them out during training. By default, only visible keypoints (visibility > 0) contribute to the loss. When this flag is enabled, invisible keypoints also participate in training, potentially improving model robustness for occluded cases.
- `--learn_truncated_kpts`: Learn position from truncated/out-of-boundary keypoints. By default, keypoints that fall outside the feature map boundary after augmentation only learn visibility (marked as invisible). When this flag is enabled, truncated keypoints also learn position information using clipped coordinates for indexing but original coordinates for offset calculation.

**ONNX export:**
- `--export_onnx`: Export model to ONNX during validation
- `--verify_onnx`: Verify ONNX output matches PyTorch
- `--dynamic_batch`: Export with dynamic batch size

## Testing / Inference

### Convert Model to ONNX
```bash
bash convert_keypoints_onnx.sh
```

This script:
1. Loads the best trained model from `exp/multi_pose/my_pose_experiment/model_best.pth`
2. Converts it to ONNX format
3. Runs inference on a test image
4. Saves visualizations and heatmaps

Uses `src/convert_to_onnx.py` for conversion and visualization.

## Code Architecture

### Directory Structure
```
src/
├── main.py                    # Main training entry point
├── test.py                    # Testing/evaluation script
├── demo.py                    # Demo inference script
├── convert_to_onnx.py         # ONNX conversion utility
└── lib/
    ├── opts.py                # All command-line options
    ├── logger.py              # Training logger
    ├── datasets/
    │   ├── dataset_factory.py # Dataset factory
    │   ├── dataset/
    │   │   ├── coco.py        # COCO dataset
    │   │   ├── coco_hp.py     # COCO human pose
    │   │   ├── kitti.py       # KITTI 3D detection
    │   │   ├── pascal.py      # Pascal VOC
    │   │   └── yolo_dataset.py # YOLO format (custom)
    │   └── sample/
    │       ├── ctdet.py       # Center detection sampler
    │       ├── multi_pose.py  # Multi-pose sampler
    │       └── ddd.py         # 3D detection sampler
    ├── detectors/
    │   ├── detector_factory.py
    │   ├── base_detector.py
    │   ├── ctdet.py           # Center detection
    │   ├── multi_pose.py      # Multi-pose detection
    │   └── ddd.py             # 3D detection
    ├── models/
    │   ├── model.py           # Model creation/loading/saving
    │   ├── data_parallel.py   # Multi-GPU support
    │   └── networks/
    │       ├── resnet_dcn.py  # ResNet with DCN
    │       ├── dlav0.py       # DLA variants
    │       ├── pose_dla_dcn.py # Pose DLA with DCN
    │       ├── large_hourglass.py
    │       └── DCNv2/         # Deformable convolutions
    ├── trains/
    │   ├── train_factory.py
    │   ├── base_trainer.py    # Base training logic
    │   ├── ctdet.py           # Center detection trainer
    │   ├── multi_pose.py      # Multi-pose trainer
    │   └── ddd.py             # 3D detection trainer
    └── utils/                 # Various utilities
```

### Task Types

CenterNet supports multiple tasks:
1. **ctdet**: Center-based object detection (2D bounding boxes)
2. **multi_pose**: Multi-person pose estimation with keypoints
3. **ddd**: 3D object detection (KITTI)
4. **exdet**: Extreme point detection

Each task has its own:
- Dataset sampler in `datasets/sample/`
- Detector in `detectors/`
- Trainer in `trains/`

### YOLO Dataset Format

The `yolo_dataset.py` implementation:
- Loads images from `data/yolo_dataset/images/{train,val,test}/`
- Loads labels from `data/yolo_dataset/labels/{train,val,test}/`
- Label format: `class_id x_center y_center width height [kp1_x kp1_y kp1_vis ...]`
- Supports 2 or 3 components per keypoint (x,y or x,y,visibility)
- Auto-detects keypoint configuration from labels
- Forces 2 classes (`_FORCED_NUM_CLASSES = 2`)

### Model Output Heads

For `multi_pose` task, the model outputs:
- `hm`: Object center heatmap (num_classes channels)
- `wh`: Bounding box width/height (2 channels)
- `reg`: Center point offset (2 channels) - if `--reg_offset`
- `hm_hp`: Keypoint heatmaps (num_joints channels) - if not `--not_hm_hp`
- `hps`: Keypoint offset from center (num_joints*2 channels)
- `hp_offset`: Keypoint local offset (2 channels) - if `--reg_hp_offset`
- `hp_vis`: Keypoint visibility logits (num_joints channels)

### Training Flow

1. `main.py` parses options via `opts.py`
2. Creates model using `models/model.py::create_model()`
3. Loads dataset via `datasets/dataset_factory.py::get_dataset()`
4. Dataset class calls `get_dataset_spec()` to configure heads
5. Creates trainer from `trains/train_factory.py`
6. Training loop calls `trainer.train()` for each epoch
7. Validation runs `trainer.val()` at intervals
8. Best model is saved and optionally converted to ONNX

### ONNX Export

The ONNX export functionality is integrated into training:
- `convert_model_to_onnx()` in `main.py` handles conversion
- Triggered when best model is saved (if `--export_onnx`)
- Can verify output consistency with `--verify_onnx`
- Exports all output heads as separate outputs
- Optionally supports dynamic batch size

## Common Commands

### Training from scratch
```bash
python src/main.py multi_pose --dataset yolo_dataset --exp_id <experiment_name> \
  --num_epochs 140 --batch_size 4 --input_h 320 --input_w 640 \
  --yolo_num_kpts 4 --input_rgb --yolo_mean "0 0 0" --yolo_std "1 1 1"
```

### Resume training
```bash
python src/main.py multi_pose --dataset yolo_dataset --exp_id <experiment_name> \
  --resume --load_model exp/multi_pose/<experiment_name>/model_last.pth
```

### Testing
```bash
python src/test.py multi_pose --dataset yolo_dataset --exp_id <experiment_name> \
  --load_model exp/multi_pose/<experiment_name>/model_best.pth \
  --input_h 320 --input_w 640 --yolo_num_kpts 4
```

### Convert to ONNX (standalone)
```bash
python src/convert_to_onnx.py multi_pose --dataset yolo_dataset \
  --load_model <path_to_pth> --arch res_18 --yolo_num_kpts 4 \
  --input_h 320 --input_w 640 --test_image <path_to_image>
```

## Dataset Setup

YOLO dataset structure:
```
data/yolo_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── classes.txt (optional, will use default if not present)
```

Label file format (one object per line):
```
class_id x_center y_center width height kp1_x kp1_y kp1_vis kp2_x kp2_y kp2_vis ...
```
- Coordinates are normalized [0,1]
- class_id is 0-indexed
- visibility: 0=not visible, 1=occluded, 2=visible (or just 0/1)

## Important Notes

- The codebase expects RGB input when `--input_rgb` is set, but OpenCV loads BGR by default. The YOLO dataset automatically handles BGR→RGB conversion when this flag is enabled.
- Mixed precision training (`--amp`) is enabled by default in training scripts for faster training.
- Output directory structure: `exp/<task>/<exp_id>/`
- Models are saved as: `model_last.pth`, `model_best.pth`, `model_<epoch>.pth`
- Debug visualizations are saved to `exp/<task>/<exp_id>/debug/` when `--val_debug_batches > 0`

### Invisible Keypoints Handling

By default, CenterNet only learns from visible keypoints (visibility > 0 in YOLO labels). Invisible keypoints are masked out and do not contribute to the loss.

**New feature (`--learn_invisible_kpts`):**
- When enabled, the model learns from both visible AND invisible keypoints
- Invisible keypoints will have their ground truth positions used for training
- This can help the model learn to predict occluded keypoint positions
- Useful when you have accurate annotations for occluded keypoints
- Located in `src/lib/datasets/sample/multi_pose.py` around line 193-262

**Implementation details:**
- Without flag: `if pts[j, 2] > 0:` - only process visible keypoints
- With flag: `if kp_is_visible or learn_invisible:` - process all keypoints
- Both visible and invisible keypoints get masks set to 1 and contribute to loss calculation
