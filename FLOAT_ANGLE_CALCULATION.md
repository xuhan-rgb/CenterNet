# 浮点坐标角度计算优化

## 修改说明

为了提高椭圆高斯热图的角度计算精度,已将关键点9的角度计算从整数坐标改为浮点坐标。

## 主要变更

### 之前的实现(整数坐标)
```python
x1, y1 = ct_int  # 使用整数坐标
x2, y2 = pt_int  # 使用整数坐标
dx = x2 - x1
dy = y2 - y1
theta = np.arctan2(dy, dx)
```

### 现在的实现(浮点坐标)
```python
# 使用浮点坐标计算角度和线段信息,获得更高的精度
x1_float, y1_float = float(ct[0]), float(ct[1])
x2_float, y2_float = float(pts[j, 0]), float(pts[j, 1])

# 计算线段方向和长度(使用浮点数)
dx_float = x2_float - x1_float
dy_float = y2_float - y1_float
segment_length = np.sqrt(dx_float**2 + dy_float**2)

# 使用浮点坐标计算线段的旋转角度
theta = np.arctan2(dy_float, dx_float)

# 椭圆中心位置也使用浮点坐标计算
ellipse_center_x = x2_float - dx_float * center_offset_ratio
ellipse_center_y = y2_float - dy_float * center_offset_ratio
```

## 优势

1. **更高的角度精度**: 使用原始浮点坐标,保留亚像素精度,避免整数量化误差
2. **更准确的椭圆方向**: 特别是对于短线段或接近45度的线段,浮点计算可以提供更准确的角度
3. **更精确的椭圆中心位置**: 椭圆中心位置的计算也使用浮点坐标,确保位置精度

## 调试输出

调试输出已更新为显示更高精度的浮点信息:
```
[KP9 DEBUG FLOAT] Center: (x.xx, y.yy), Keypoint: (x.xx, y.yy)
[KP9 DEBUG FLOAT] dx=x.xxxx, dy=y.xxxx, seg_len=z.zzzz
[KP9 DEBUG FLOAT] theta=t.tttttt rad = angle.ddd°
[KP9 DEBUG FLOAT] Ellipse center: (x.xxxx, y.yyyy)
```

## 测试方法

运行训练并观察调试输出:
```bash
bash train_keypoints.sh
```

查看生成的热图可视化文件,确认椭圆方向正确。

## 文件修改

- `src/lib/datasets/sample/multi_pose.py`: 关键点9的椭圆高斯绘制逻辑(第295-366行)
