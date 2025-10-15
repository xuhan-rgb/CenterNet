# Objects as Points
Object detection, 3D detection, and pose estimation using center point detection:
![](readme/fig2.png)
> [**Objects as Points**](http://arxiv.org/abs/1904.07850),            
> Xingyi Zhou, Dequan Wang, Philipp Kr&auml;henb&uuml;hl,        
> *arXiv technical report ([arXiv 1904.07850](http://arxiv.org/abs/1904.07850))*         


Contact: [zhouxy2017@gmail.com](mailto:zhouxy2017@gmail.com). Any questions or discussions are welcomed! 


# JSZR update
1. 增加了yolo格式的inference代码，增加了训练的相关配置。
2. 训练 bash train_keypoints.sh
3. 测试 bash convert_keypoints_onnx.sh
4. 训练脚本示例新增 `--yolo_force_num_classes`、`--yolo_dataset_dir` 与 `--keep_bbox_without_kpts` 标志，可直接指定类别数、数据集路径并在全部关键点不可见时仍保留框回归（见 `train_keypoints.sh`）。
5. `data/test_data/coco_to_yolo.py` 自动生成类别映射，执行 `source ~/anaconda3/bin/activate && conda activate py310 && python data/test_data/coco_to_yolo.py --coco_json <输入COCO标注> --output_dir <输出目录>` 直接完成转换。
6. 保存数据增强后的图像、框、关键点与热力图：在训练或数据调试命令中追加标志 `--debug_aug_vis`，例如 `source ~/anaconda3/bin/activate && conda activate py310 && python src/main.py multi_pose --exp_id debug_aug --debug_aug_vis ...`。增强样本将保存到 `tmp/`，包含边框/关键点叠加图、检测热力图与关键点热力图（含原图叠加）。
7. `test_case/infer_onnx_runtime.py` 现支持直接传入图片目录：`python infer_onnx_runtime.py --config onnx_infer.yaml path/to/dir/` 会自动批量处理目录下常见格式图片，并按文件名输出可视化/JSON。
8. `src/convert_to_onnx.py` 在推理测试时可指定 `--test_label` 结合 `--eval_iou_thresh` 计算检测准确率与关键点平均偏差，输出到日志中便于核对。
9. `test_case/eval_onnx_dataset.py` 可对 `data/yolo_annotations` 等数据集的 `train`/`val` 子集批量评估ONNX模型精度，输出可见度为0/1的平均偏差与准确率：`python eval_onnx_dataset.py --config onnx_infer.yaml --dataset_root data/yolo_annotations/ --splits train val`。

## Updates

 - (June, 2020) We released a state-of-the-art Lidar-based 3D detection and tracking framework [CenterPoint](https://github.com/tianweiy/CenterPoint).
 - (April, 2020) We released a state-of-the-art (multi-category-/ pose-/ 3d-) tracking extension [CenterTrack](https://github.com/xingyizhou/CenterTrack).

## Abstract 

Detection identifies objects as axis-aligned boxes in an image. Most successful object detectors enumerate a nearly exhaustive list of potential object locations and classify each. This is wasteful, inefficient, and requires additional post-processing. In this paper, we take a different approach. We model an object as a single point -- the center point of its bounding box. Our detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and even pose. Our center point based approach, CenterNet, is end-to-end differentiable, simpler, faster, and more accurate than corresponding bounding box based detectors. CenterNet achieves the best speed-accuracy trade-off on the MS COCO dataset, with 28.1% AP at 142 FPS, 37.4% AP at 52 FPS, and 45.1% AP with multi-scale testing at 1.4 FPS. We use the same approach to estimate 3D bounding box in the KITTI benchmark and human pose on the COCO keypoint dataset. Our method performs competitively with sophisticated multi-stage methods and runs in real-time.

## Highlights

- **Simple:** One-sentence method summary: use keypoint detection technic to detect the bounding box center point and regress to all other object properties like bounding box size, 3d information, and pose.

- **Versatile:** The same framework works for object detection, 3d bounding box estimation, and multi-person pose estimation with minor modification.

- **Fast:** The whole process in a single network feedforward. No NMS post processing is needed. Our DLA-34 model runs at *52* FPS with *37.4* COCO AP.

- **Strong**: Our best single model achieves *45.1*AP on COCO test-dev.

- **Easy to use:** We provide user friendly testing API and webcam demos.

## Main results

### Object Detection on COCO validation

| Backbone     |  AP / FPS | Flip AP / FPS|  Multi-scale AP / FPS |
|--------------|-----------|--------------|-----------------------|
|Hourglass-104 | 40.3 / 14 | 42.2 / 7.8   | 45.1 / 1.4            |
|DLA-34        | 37.4 / 52 | 39.2 / 28    | 41.7 / 4              |
|ResNet-101    | 34.6 / 45 | 36.2 / 25    | 39.3 / 4              |
|ResNet-18     | 28.1 / 142| 30.0 / 71    | 33.2 / 12             |

### Keypoint detection on COCO validation

| Backbone     |  AP       |  FPS         |
|--------------|-----------|--------------|
|Hourglass-104 | 64.0      |    6.6       |
|DLA-34        | 58.9      |    23        |

### 3D bounding box detection on KITTI validation

|Backbone|FPS|AP-E|AP-M|AP-H|AOS-E|AOS-M|AOS-H|BEV-E|BEV-M|BEV-H| 
|--------|---|----|----|----|-----|-----|-----|-----|-----|-----|
|DLA-34  |32 |96.9|87.8|79.2|93.9 |84.3 |75.7 |34.0 |30.5 |26.8 |


All models and details are available in our [Model zoo](readme/MODEL_ZOO.md).

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Use CenterNet

We support demo for image/ image folder, video, and webcam. 

First, download the models (By default, [ctdet_coco_dla_2x](https://drive.google.com/file/d/18Q3fzzAsha_3Qid6mn4jcIFPeOGUaj1d) for detection and 
[multi_pose_dla_3x](https://drive.google.com/file/d/1mC2PAQT_RuHi_9ZMZgkt4rg7BSY2_Lkd) for human pose estimation) 
from the [Model zoo](readme/MODEL_ZOO.md) and put them in `CenterNet_ROOT/models/`.

For object detection on images/ video, run:

~~~
python demo.py ctdet --demo /path/to/image/or/folder/or/video --load_model ../models/ctdet_coco_dla_2x.pth
~~~
We provide example images in `CenterNet_ROOT/images/` (from [Detectron](https://github.com/facebookresearch/Detectron/tree/master/demo)). If set up correctly, the output should look like

<p align="center"> <img src='readme/det1.png' align="center" height="230px"> <img src='readme/det2.png' align="center" height="230px"> </p>

For webcam demo, run     

~~~
python demo.py ctdet --demo webcam --load_model ../models/ctdet_coco_dla_2x.pth
~~~

Similarly, for human pose estimation, run:

~~~
python demo.py multi_pose --demo /path/to/image/or/folder/or/video/or/webcam --load_model ../models/multi_pose_dla_3x.pth
~~~
The result for the example images should look like:

<p align="center">  <img src='readme/pose1.png' align="center" height="200px"> <img src='readme/pose2.png' align="center" height="200px"> <img src='readme/pose3.png' align="center" height="200px">  </p>

You can add `--debug 2` to visualize the heatmap outputs.
You can add `--flip_test` for flip test.

To use this CenterNet in your own project, you can 

~~~
import sys
CENTERNET_PATH = /path/to/CenterNet/src/lib/
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts

MODEL_PATH = /path/to/model
TASK = 'ctdet' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)

img = image/or/path/to/your/image/
ret = detector.run(img)['results']
~~~
`ret` will be a python dict: `{category_id : [[x1, y1, x2, y2, score], ...], }`

## Benchmark Evaluation and Training

After [installation](readme/INSTALL.md), follow the instructions in [DATA.md](readme/DATA.md) to setup the datasets. Then check [GETTING_STARTED.md](readme/GETTING_STARTED.md) to reproduce the results in the paper.
We provide scripts for all the experiments in the [experiments](experiments) folder.

## Develop

If you are interested in training CenterNet in a new dataset, use CenterNet in a new task, or use a new network architecture for CenterNet, please refer to [DEVELOP.md](readme/DEVELOP.md). Also feel free to send us emails for discussions or suggestions.

## Third-party resources

- CenterNet + embedding learning based tracking: [FairMOT](https://github.com/ifzhang/FairMOT) from [Yifu Zhang](https://github.com/ifzhang).
- Detectron2 based implementation: [CenterNet-better](https://github.com/FateScript/CenterNet-better) from [Feng Wang](https://github.com/FateScript).
- Keras Implementation: [keras-centernet](https://github.com/see--/keras-centernet) from [see--](https://github.com/see--) and [keras-CenterNet](https://github.com/xuannianz/keras-CenterNet) from [xuannianz](https://github.com/xuannianz).
- MXnet implementation: [mxnet-centernet](https://github.com/Guanghan/mxnet-centernet) from [Guanghan Ning](https://github.com/Guanghan).
- Stronger human open estimation models: [centerpose](https://github.com/tensorboy/centerpose) from [tensorboy](https://github.com/tensorboy).
- TensorRT extension with ONNX models: [TensorRT-CenterNet](https://github.com/CaoWGG/TensorRT-CenterNet) from [Wengang Cao](https://github.com/CaoWGG).
- CenterNet + DeepSORT tracking implementation: [centerNet-deep-sort](https://github.com/kimyoon-young/centerNet-deep-sort) from [kimyoon-young](https://github.com/kimyoon-young).
- Blogs on training CenterNet on custom datasets (in Chinese): [ships](https://blog.csdn.net/weixin_42634342/article/details/97756458) from [Rhett Chen](https://blog.csdn.net/weixin_42634342) and [faces](https://blog.csdn.net/weixin_41765699/article/details/100118353) from [linbior](https://me.csdn.net/weixin_41765699).


## License

CenterNet itself is released under the MIT License (refer to the LICENSE file for details).
Portions of the code are borrowed from [human-pose-estimation.pytorch](https://github.com/Microsoft/human-pose-estimation.pytorch) (image transform, resnet), [CornerNet](https://github.com/princeton-vl/CornerNet) (hourglassnet, loss functions), [dla](https://github.com/ucbdrive/dla) (DLA network), [DCNv2](https://github.com/CharlesShang/DCNv2)(deformable convolutions), [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)(Pascal VOC evaluation) and [kitti_eval](https://github.com/prclibo/kitti_eval) (KITTI dataset evaluation). Please refer to the original License of these projects (See [NOTICE](NOTICE)).

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @inproceedings{zhou2019objects,
      title={Objects as Points},
      author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
      booktitle={arXiv preprint arXiv:1904.07850},
      year={2019}
    }
