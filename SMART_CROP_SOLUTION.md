# 避免Keypoint截断的解决方案

## 问题总结

数据增强（random crop/scale）可能导致keypoint超出输出特征图边界(0-160, 0-80)，造成训练信息丢失。

从测试结果看：
- **Random Crop**: 74.07% keypoints被截断
- **Smart Crop (第一版)**: 99.57% keypoints被截断 ❌ (实现有bug)

## 根本原因

affine变换的数学特性导致很难预先保证所有keypoint都在边界内，因为：
1. 变换矩阵由crop center、scale、rotation共同决定
2. 变换是非线性的，点的映射关系复杂
3. 即使原始点在安全区域，变换后也可能超出边界

## 推荐解决方案

### 方案1: 使用 `--not_rand_crop` (最简单) ⭐

**完全禁用random crop，使用温和的数据增强**：

```bash
python src/main.py multi_pose --dataset yolo_dataset \
  --not_rand_crop \          # 禁用random crop
  --scale 0.2 \               # 温和的缩放 (默认0.4)
  --shift 0.05 \              # 温和的平移 (默认0.1)
  --flip 0.5 \                # 保留水平翻转
  --input_h 320 --input_w 640 \
  --yolo_num_kpts 10 \
  --learn_invisible_kpts \
  ...
```

**优点**:
- ✅ 简单可靠
- ✅ 不会出现keypoint截断
- ✅ 仍保留flip和温和的scale/shift增强

**缺点**:
- ⚠️ 数据增强多样性降低
- ⚠️ 可能影响模型泛化能力(需要实验验证)

### 方案2: 降低Random Crop的激进程度

**调整crop参数，减少极端情况**：

修改 `src/lib/datasets/sample/multi_pose.py:113-117`：

```python
# 原始代码
s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))  # 缩放范围: 0.6x - 1.4x

# 修改为更温和的范围
s = s * np.random.choice(np.arange(0.8, 1.2, 0.1))  # 缩放范围: 0.8x - 1.2x
```

**优点**:
- ✅ 保留random crop的多样性
- ✅ 减少截断概率

**缺点**:
- ⚠️ 仍有一定概率出现截断
- ⚠️ 需要修改源代码

### 方案3: 使用 `--learn_truncated_kpts`

**让模型从截断的keypoint中学习**：

```bash
python src/main.py multi_pose --dataset yolo_dataset \
  --learn_truncated_kpts \    # 截断的点也参与训练
  --learn_invisible_kpts \    # 不可见的点也参与训练
  ...
```

**优点**:
- ✅ 充分利用所有数据
- ✅ 模型学习处理边界情况

**缺点**:
- ⚠️ 截断点的真实坐标在边界外，可能影响学习质量
- ⚠️ 需要实验验证效果

### 方案4: 后处理过滤

**在数据加载时检测并跳过截断严重的样本**：

修改 `multi_pose.py`，在变换后检查所有keypoint，如果超过一定比例截断就重新采样。

**优点**:
- ✅ 保证训练数据质量
- ✅ 不影响数据增强多样性

**缺点**:
- ⚠️ 实现复杂
- ⚠️ 可能降低训练速度(重采样)

## 最佳实践

### 推荐配置1: 保守型 (推荐)

```bash
python src/main.py multi_pose --dataset yolo_dataset \
  --exp_id safe_training \
  --not_rand_crop \              # 禁用激进crop
  --scale 0.15 \                  # 温和缩放
  --shift 0.05 \                  # 温和平移
  --flip 0.5 \                    # 保留flip
  --no_color_aug \                # 可选：禁用颜色增强
  --input_h 320 --input_w 640 \
  --yolo_num_kpts 10 \
  --yolo_flip_pairs "0-1,2-3,4-5,6-7" \
  --input_rgb \
  --yolo_mean "0 0 0" \
  --yolo_std "1 1 1" \
  --learn_invisible_kpts \
  --num_epochs 140 \
  --batch_size 4 \
  --lr 1.25e-4
```

**适用场景**:
- keypoint精度要求高
- 不能容忍截断
- 数据集质量好，标注准确

### 推荐配置2: 平衡型

```bash
python src/main.py multi_pose --dataset yolo_dataset \
  --exp_id balanced_training \
  # 使用默认random crop，但配合learn_truncated
  --learn_truncated_kpts \
  --learn_invisible_kpts \
  --flip 0.5 \
  --aug_rot 0.05 \                # 添加小角度旋转
  --input_h 320 --input_w 640 \
  ...
```

**适用场景**:
- 需要强数据增强
- 可以接受一定的截断
- 数据集较小，需要增强泛化能力

## 训练监控

无论选择哪种方案，建议监控以下指标：

1. **截断率**: 使用 `test_smart_crop.py` 定期检查
2. **Keypoint可见性损失**: 观察 `hp_vis_loss`
3. **Keypoint位置损失**: 观察 `hp_loss`
4. **验证集表现**: 重点关注边缘物体的检测

## 结论

**对于当前任务（充电桩keypoint检测）**：

由于：
- 输出分辨率较低 (160x80)
- Keypoint数量较多 (10个)
- 精度要求高

**建议使用方案1**: `--not_rand_crop` + 温和的scale/shift

这是最可靠的方案，避免了所有截断问题。如果后续发现泛化能力不足，再考虑方案2或方案3。

---

## 快速开始

### 1. 清理调试代码

当前代码中有很多调试日志，建议清理：

```bash
# 恢复干净的代码
git diff src/lib/datasets/sample/multi_pose.py
git checkout src/lib/datasets/sample/multi_pose.py  # 或手动清理
```

### 2. 开始训练

```bash
# 使用推荐配置
bash train_keypoints.sh  # 修改脚本，添加 --not_rand_crop
```

### 3. 验证效果

```bash
# 测试模型
bash convert_keypoints_onnx.sh

# 检查推理结果
python test_case/infer_onnx_runtime.py
```

---
生成时间: 2025-10-15
文档作者: Claude Code
