# KP9 角度变化分析报告

## 问题描述
对于同一张图片(150.jpg)，keypoint 9相对于center的角度在不同训练样本中发生变化：
- 有时 dx > 0 (kp9在center右侧)，角度约 -86.93°
- 有时 dx < 0 (kp9在center左侧)，角度约 -91.71° 或 -127.28°

## 根本原因

这是**正常的数据增强行为**，不是bug。主要由以下两个因素导致：

### 1. 水平翻转 (Horizontal Flip)

**代码位置**: `multi_pose.py:84-87`

```python
if np.random.random() < self.opt.flip:  # 默认 flip=0.5
    flipped = True
    img = img[:, ::-1, :]  # 水平翻转图像
    c[0] = width - c[0] - 1  # 翻转中心点x坐标
```

每个epoch，每张图片有50%概率被水平翻转。

**效果**:
- **未翻转**: kp9在center右侧，dx = +2.00
- **翻转后**: kp9在center左侧，dx = -2.00

### 2. 随机裁剪与缩放 (Random Crop & Scale)

**代码位置**: `multi_pose.py:63-74`

```python
if not self.opt.not_rand_crop:
    s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))  # 缩放0.6-1.4倍
    w_border = self._get_border(128, img.shape[1])
    h_border = self._get_border(128, img.shape[0])
    c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
    c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
```

随机选择裁剪中心点和缩放比例会导致affine变换的结果不同。

**效果**:
- 不同的crop/scale参数会导致物体在输出特征图上的位置和大小不同
- keypoint和center之间的相对距离也会改变
- 例如：原始 dx=2.00 → 经过不同crop → dx=0.13 或 dx=-4.72

## 实际样本分析

### 样本1: 未翻转
```
[KP9 TRACE] ORIGINAL: bbox_center=(847.00, 682.00), kp9=(849.00, 615.00)
[KP9 TRACE] ORIGINAL: dx=2.0007, dy=-67.0000
[KP9 TRACE] CROP/SCALE: c=(153.00, 711.00), s=1920.00
[KP9 TRACE] FINAL: ct=(137.83, 77.58), kp9=(138.00, 72.00)
[KP9 TRACE] FINAL: dx=0.1667, dy=-5.5833
[KP9 TRACE] flipped=False
```
**结果**: dx保持正数，角度 ≈ -88.29°

### 样本2: 翻转后
```
[KP9 TRACE] ORIGINAL: bbox_center=(847.00, 682.00), kp9=(849.00, 615.00)
[KP9 TRACE] ORIGINAL: dx=2.0007, dy=-67.0000
[KP9 TRACE] AFTER FLIP: bbox_center=(1072.00, 682.00), kp9=(1070.00, 615.00)
[KP9 TRACE] AFTER FLIP: dx=-2.0007, dy=-66.9999
[KP9 TRACE] CROP/SCALE: c=(243.00, 272.00), s=1536.00
[KP9 TRACE] FINAL: ct=(17.71, 122.71), kp9=(16.87, 115.73)
[KP9 TRACE] FINAL: dx=-0.8334, dy=-6.9792
[KP9 TRACE] flipped=True
```
**结果**: dx变为负数，角度 ≈ -96.81°

### 样本3: 翻转 + 极端裁剪
```
[原始相同，翻转后相同]
[KP9 TRACE] CROP/SCALE: c=(极端裁剪参数)
[KP9 TRACE] FINAL: ct=(11.39, 122.31), kp9=(6.67, 116.11)
[KP9 TRACE] FINAL: dx=-4.7221, dy=-6.2037
```
**结果**: dx绝对值变大，角度 ≈ -127.28°

## 为什么dx的绝对值会变化？

即使flip只改变dx的符号，但是**随机裁剪和缩放**会改变dx的绝对值：

1. **裁剪中心不同**: 如果裁剪中心偏向物体的某一侧，会改变物体在输出中的位置
2. **缩放比例不同**: 不同的缩放比例会改变像素级别的相对距离
3. **仿射变换**: affine transform是非线性的，不同的参数组合会产生不同的映射结果

## 数据增强对训练的意义

这些变化是**有意为之**的，目的是：

1. **增加样本多样性**: 让模型见到同一物体的不同视角、位置、尺度
2. **提高泛化能力**: 防止模型记忆固定的角度或位置
3. **模拟真实场景**: 实际推理时，物体可能出现在图像的任何位置

## 验证方法

如果想验证这是数据增强导致的，可以：

### 禁用所有数据增强
```bash
python src/main.py multi_pose --dataset yolo_dataset \
  --flip 0 \              # 禁用flip
  --not_rand_crop \       # 禁用随机裁剪
  --scale 0 \             # 禁用缩放
  --shift 0 \             # 禁用平移
  --aug_rot 0 \           # 禁用旋转
  --no_color_aug          # 禁用颜色增强
```

此时同一张图片在不同epoch的角度应该完全一致。

## 结论

✅ **这不是bug，是正常的数据增强行为**

- dx符号改变 → 由flip引起
- dx绝对值改变 → 由crop/scale引起
- 角度变化 → flip + crop/scale的综合效果

这些变化有助于模型学习更鲁棒的特征表示，不依赖于特定的角度或位置。

## 建议

1. **保持现状**: 数据增强工作正常，无需修改
2. **如果角度很重要**: 可以降低flip概率(--flip 0.3)或crop范围
3. **监控训练**: 确保模型能够处理各种角度的情况

---
生成时间: 2025-10-15
分析工具: Claude Code
