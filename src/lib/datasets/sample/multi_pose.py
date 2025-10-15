from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math


class MultiPoseDataset(data.Dataset):

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):

        import time

        item_timestamp = time.time()
        item_tag = f"idx={index} ts={item_timestamp:.3f}"

        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]["file_name"]
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)
        if getattr(self.opt, "input_rgb", False):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        debug_aug_vis = bool(getattr(self.opt, "debug_aug_vis", False))
        sample_debug = bool(getattr(self.opt, "log_sample_details", False))
        if debug_aug_vis and not hasattr(self, "_debug_aug_tmp_dir"):
            tmp_dir = os.path.join(getattr(self.opt, "root_dir", "."), "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            self._debug_aug_tmp_dir = tmp_dir

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2.0, img.shape[0] / 2.0], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        rot = 0

        flipped = False
        if self.split == "train":
            # 新增选项: 智能crop，确保所有keypoint在边界内
            use_smart_crop = getattr(self.opt, "smart_crop", False)

            if use_smart_crop:
                # 智能Crop策略: 基于所有object的keypoint计算安全的crop范围
                # 收集所有可见keypoint的坐标
                all_kpts = []
                for ann in anns:
                    pts = np.array(ann["keypoints"], np.float32).reshape(-1, 3)
                    visible_pts = pts[pts[:, 2] > 0, :2]  # 只考虑可见的keypoint
                    if len(visible_pts) > 0:
                        all_kpts.append(visible_pts)

                if len(all_kpts) > 0:
                    all_kpts = np.concatenate(all_kpts, axis=0)
                    # 计算所有keypoint的边界
                    kpt_min_x, kpt_min_y = all_kpts.min(axis=0)
                    kpt_max_x, kpt_max_y = all_kpts.max(axis=0)

                    # 随机缩放比例
                    scale_factor = np.random.choice(np.arange(0.8, 1.3, 0.1))
                    s = s * scale_factor

                    # 计算crop后的输出范围（考虑缩放）
                    # affine变换: output_size / s 对应原图中的范围
                    crop_size_in_img = s  # 原图中被crop的区域大小

                    # 为了确保所有keypoint都在输出范围内，crop中心需要满足：
                    # kpt_min, kpt_max 都要在 [c - crop_size/2, c + crop_size/2] 范围内
                    c_min_x = kpt_max_x - crop_size_in_img / 2.0
                    c_max_x = kpt_min_x + crop_size_in_img / 2.0
                    c_min_y = kpt_max_y - crop_size_in_img / 2.0
                    c_max_y = kpt_min_y + crop_size_in_img / 2.0

                    # 限制在图像边界内
                    c_min_x = max(crop_size_in_img / 2.0, c_min_x)
                    c_max_x = min(width - crop_size_in_img / 2.0, c_max_x)
                    c_min_y = max(crop_size_in_img / 2.0, c_min_y)
                    c_max_y = min(height - crop_size_in_img / 2.0, c_max_y)

                    # 如果范围有效，随机选择crop中心
                    if c_max_x > c_min_x and c_max_y > c_min_y:
                        c[0] = np.random.uniform(c_min_x, c_max_x)
                        c[1] = np.random.uniform(c_min_y, c_max_y)
                    # 否则保持原始中心（使用图像中心）
                else:
                    # 没有可见keypoint，使用原始策略
                    s = s * np.random.choice(np.arange(0.8, 1.3, 0.1))

            elif not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.aug_rot:
                rf = self.opt.rotate
                rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)

            if np.random.random() < self.opt.flip:  # 去除这个数据增强
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(c, s, rot, [self.opt.input_res, self.opt.input_res])
        inp = cv2.warpAffine(img, trans_input, (self.opt.input_res, self.opt.input_res), flags=cv2.INTER_LINEAR)
        debug_image = inp.copy() if debug_aug_vis else None
        inp = inp.astype(np.float32) / 255.0
        if self.split == "train" and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_res = self.opt.output_res
        num_joints = self.num_joints
        trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
        trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

        hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)
        hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
        dense_kps = np.zeros((num_joints, 2, output_res, output_res), dtype=np.float32)
        dense_kps_mask = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)
        hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
        hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
        hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)
        hp_vis = np.zeros((self.max_objs, num_joints), dtype=np.float32)
        hp_vis_mask = np.zeros((self.max_objs, num_joints), dtype=np.float32)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

        debug_keypoints = [] if debug_aug_vis and num_joints > 0 else None
        debug_boxes = [] if debug_aug_vis else None
        debug_kp9 = bool(getattr(self.opt, "debug_kp9", False))
        gt_det = []
        keep_bbox_without_kpts = bool(getattr(self.opt, "keep_bbox_without_kpts", False))
        for k in range(num_objs):
            ann = anns[k]
            raw_bbox_xywh = np.array(ann["bbox"], dtype=np.float32)
            bbox = self._coco_box_to_bbox(raw_bbox_xywh)
            cls_id = int(ann["category_id"]) - 1
            cls_name = (
                self.class_names[cls_id]
                if isinstance(getattr(self, "class_names", None), list) and 0 <= cls_id < len(self.class_names)
                else f"class_{cls_id}"
            )
            pts = np.array(ann["keypoints"], np.float32).reshape(num_joints, 3)
            pts_for_input = pts.copy() if debug_keypoints is not None else None
            kp9_orig = None
            bbox_center_x = None
            if debug_kp9 and num_joints > 9:
                kp9_orig = pts[9, :2].copy()
                bbox_center_x = raw_bbox_xywh[0] + raw_bbox_xywh[2] / 2.0

            if sample_debug and k == 0 and num_joints > 0:
                first_joint = pts[0]
                print(
                    "[multi_pose sample] {} img={} label cls={} ({}) bbox_xywh=({:.2f}, {:.2f}, {:.2f}, {:.2f}), first_joint=({:.2f}, {:.2f}), vis={:.1f}".format(
                        item_tag,
                        file_name,
                        cls_id,
                        cls_name,
                        raw_bbox_xywh[0],
                        raw_bbox_xywh[1],
                        raw_bbox_xywh[2],
                        raw_bbox_xywh[3],
                        first_joint[0],
                        first_joint[1],
                        first_joint[2],
                    )
                )
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                pts[:, 0] = width - pts[:, 0] - 1
                for e in self.flip_idx:
                    pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()

                if sample_debug and k == 0 and num_joints > 0:
                    print(
                        "[multi_pose sample] {} after flip first_joint=({:.2f}, {:.2f}), vis={:.1f}".format(
                            item_tag, pts[0, 0], pts[0, 1], pts[0, 2]
                        )
                    )

            if debug_boxes is not None:
                bbox_vis = bbox.copy()
                bbox_vis[:2] = affine_transform(bbox_vis[:2], trans_input)
                bbox_vis[2:] = affine_transform(bbox_vis[2:], trans_input)
                bbox_vis[[0, 2]] = np.clip(bbox_vis[[0, 2]], 0, self.opt.input_res - 1)
                bbox_vis[[1, 3]] = np.clip(bbox_vis[[1, 3]], 0, self.opt.input_res - 1)
                debug_boxes.append(
                    {
                        "bbox": bbox_vis,
                        "cls_id": int(cls_id),
                        "cls_name": cls_name,
                    }
                )

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox = np.clip(bbox, 0, output_res - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if (h > 0 and w > 0) or (rot != 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                if self.opt.hm_gauss > 0:
                    radius = max(0, int(round(self.opt.hm_gauss)))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                if sample_debug and k == 0:
                    print(
                        "[multi_pose sample] {} bbox_affine=({:.2f}, {:.2f}, {:.2f}, {:.2f}), center=({:.2f}, {:.2f}), ct_int=({}, {}), wh=({:.2f}, {:.2f}), rot={:.2f}".format(
                            item_tag,
                            bbox[0],
                            bbox[1],
                            bbox[2],
                            bbox[3],
                            ct[0],
                            ct[1],
                            ct_int[0],
                            ct_int[1],
                            w,
                            h,
                            rot,
                        )
                    )
                wh[k] = 1.0 * w, 1.0 * h
                ind[k] = ct_int[1] * output_res + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                visible_before_aug = np.sum(pts[:, 2] > 0)
                if visible_before_aug == 0:
                    hm[cls_id, ct_int[1], ct_int[0]] = max(hm[cls_id, ct_int[1], ct_int[0]], 0.9999)
                    if not keep_bbox_without_kpts:
                        reg_mask[k] = 0

                hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                hp_radius = max(0, int(hp_radius))
                if self.opt.hm_gauss > 0:
                    hp_radius = max(0, int(round(self.opt.hm_gauss)))
                visible_after_aug = 0
                learn_invisible = getattr(self.opt, "learn_invisible_kpts", False)
                learn_truncated = getattr(self.opt, "learn_truncated_kpts", False)
                for j in range(num_joints):
                    debug_entry = None
                    if debug_keypoints is not None and pts_for_input is not None and j < pts_for_input.shape[0]:
                        kp_inp = affine_transform(pts_for_input[j, :2], trans_input)
                        debug_entry = [float(kp_inp[0]), float(kp_inp[1]), float(pts[j, 2]), int(j)]

                    # 跳过属性为-1的关键点（不参与任何损失计算，包括可见性损失）
                    if pts[j, 2] < 0:
                        pts[j, 2] = 0  # 标记为不可见
                        hp_vis[k, j] = 0.0
                        hp_vis_mask[k, j] = 0.0  # 不参与可见性损失计算
                        if debug_entry is not None:
                            debug_entry[2] = float(pts[j, 2])
                            debug_keypoints.append(debug_entry)
                        continue

                    # 对于有效的关键点，设置可见性mask为1
                    hp_vis_mask[k, j] = 1.0

                    # 对于不可见的关键点，如果启用learn_invisible_kpts，也进行变换
                    kp_is_visible = pts[j, 2] > 0
                    if kp_is_visible or learn_invisible:
                        # 记录kp9的affine变换
                        if debug_kp9 and kp9_orig is not None and k < 3 and j == 9:
                            kp9_before_kp_affine = pts[j, :2].copy()

                        pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)

                        # 检查是否在边界内
                        is_in_bounds = 0 <= pts[j, 0] < output_res and 0 <= pts[j, 1] < output_res

                        # 可选的kp9调试：仅保存必要信息
                        if debug_kp9 and kp9_orig is not None and k < 3 and j == 9:
                            kp9_affine_info = {
                                "before": kp9_before_kp_affine.copy(),
                                "after": pts[j, :2].copy(),
                                "ct": ct.copy(),
                                "flipped": bool(flipped),
                                "in_bounds": bool(is_in_bounds),
                            }
                            if not flipped and is_in_bounds and bbox_center_x is not None:
                                if kp9_orig[0] > bbox_center_x and pts[j, 0] < ct[0]:
                                    kp9_affine_info["dx_warning"] = True

                        if is_in_bounds:
                            # 在边界内的点
                            if kp_is_visible:
                                visible_after_aug += 1

                            # 对于不可见的点，如果learn_invisible启用，也设置mask和offset
                            if kp_is_visible or learn_invisible:
                                kps[k, j * 2 : j * 2 + 2] = pts[j, :2] - ct_int
                                kps_mask[k, j * 2 : j * 2 + 2] = 1
                                pt_int = pts[j, :2].astype(np.int32)

                                hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
                                hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
                                hp_mask[k * num_joints + j] = 1
                                if sample_debug and k == 0 and j == 0:
                                    print(
                                        "[multi_pose sample] {} first joint after affine=({:.4f}, {:.4f}), "
                                        "center_int=({:.4f}, {:.4f}), offset=({:.4f}, {:.4f}), vis={:.1f}, in_bounds=True".format(
                                            item_tag,
                                            pts[j, 0],
                                            pts[j, 1],
                                            ct_int[0],
                                            ct_int[1],
                                            pts[j, 0] - ct_int[0],
                                            pts[j, 1] - ct_int[1],
                                            pts[j, 2],
                                        )
                                    )
                                    print(
                                        "[multi_pose sample] {} first hp_offset=({:.4f}, {:.4f}), hp_ind={}, hp_mask={}".format(
                                            item_tag,
                                            hp_offset[k * num_joints + j, 0],
                                            hp_offset[k * num_joints + j, 1],
                                            int(hp_ind[k * num_joints + j]),
                                            int(hp_mask[k * num_joints + j]),
                                        )
                                    )
                                if self.opt.dense_hp:
                                    # must be before draw center hm gaussian
                                    draw_dense_reg(
                                        dense_kps[j], hm[cls_id], ct_int, pts[j, :2] - ct_int, radius, is_offset=True
                                    )
                                    draw_gaussian(dense_kps_mask[j], ct_int, radius)

                                # 对于可见的点或者启用learn_invisible的不可见点，都绘制heatmap
                                if kp_is_visible or learn_invisible:
                                    # 先保证关键点自身位置存在峰值
                                    draw_gaussian(hm_hp[j], pt_int, hp_radius)

                                    # if j == 9:
                                    #     # 额外沿中心点到关键点的线段采样，补充靠近关键点的热力值
                                    #     x1, y1 = ct_int
                                    #     x2, y2 = pt_int
                                    #     num_samples = max(abs(x2 - x1), abs(y2 - y1)) + 1
                                    #     if num_samples > 1:
                                    #         x_points = np.linspace(x1, x2, num=num_samples, dtype=int)
                                    #         y_points = np.linspace(y1, y2, num=num_samples, dtype=int)
                                    #         line_radius = max(1, hp_radius // 2)
                                    #         tail_len = max(1, min(num_samples, int(np.ceil(num_samples * 0.5))))
                                    #         for x_line, y_line in zip(x_points[-tail_len:], y_points[-tail_len:]):
                                    #             if 0 <= x_line < output_res and 0 <= y_line < output_res:
                                    #                 draw_gaussian(hm_hp[j], (x_line, y_line), line_radius)

                        else:
                            # 超出边界的点（截断的点）
                            # 无论原本可见或不可见，超出边界都标记为不可见
                            pts[j, 2] = 0

                            # 如果启用learn_truncated_kpts，截断的点也参与位置学习
                            if learn_truncated:
                                # Clip坐标到边界内用于mask设置
                                pts_clipped = np.array(
                                    [np.clip(pts[j, 0], 0, output_res - 1), np.clip(pts[j, 1], 0, output_res - 1)]
                                )

                                kps[k, j * 2 : j * 2 + 2] = pts[j, :2] - ct_int
                                kps_mask[k, j * 2 : j * 2 + 2] = 1
                                pt_int = pts_clipped.astype(np.int32)

                                hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
                                hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
                                hp_mask[k * num_joints + j] = 1

                                if sample_debug and k == 0 and j == 0:
                                    print(
                                        "[multi_pose sample] {} TRUNCATED first joint after affine=({:.4f}, {:.4f}), "
                                        "clipped=({:.4f}, {:.4f}), center_int=({:.4f}, {:.4f}), vis={:.1f}, in_bounds=False".format(
                                            item_tag,
                                            pts[j, 0],
                                            pts[j, 1],
                                            pts_clipped[0],
                                            pts_clipped[1],
                                            ct_int[0],
                                            ct_int[1],
                                            pts[j, 2],
                                        )
                                    )
                    else:
                        # 不可见且不学习的情况
                        pts[j, 2] = 0

                    hp_vis[k, j] = 1.0 if pts[j, 2] > 0 else 0.0
                    if debug_entry is not None:
                        debug_entry[2] = float(pts[j, 2])
                        debug_keypoints.append(debug_entry)
                if num_joints > 0 and visible_before_aug > 0 and visible_after_aug == 0 and not keep_bbox_without_kpts:
                    hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
                    reg_mask[k] = 0
                draw_gaussian(hm[cls_id], ct_int, radius)
                gt_det.append(
                    [ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2, 1]
                    + pts[:, :2].reshape(num_joints * 2).tolist()
                    + pts[:, 2].tolist()
                    + [cls_id]
                )

        if rot != 0:
            hm = hm * 0 + 0.9999
            reg_mask *= 0
            kps_mask *= 0
        ret = {
            "input": inp,
            "hm": hm,
            "reg_mask": reg_mask,
            "ind": ind,
            "wh": wh,
            "hps": kps,
            "hps_mask": kps_mask,
            "hp_vis": hp_vis,
            "hp_vis_mask": hp_vis_mask,
        }
        if self.opt.dense_hp:
            dense_kps = dense_kps.reshape(num_joints * 2, output_res, output_res)
            dense_kps_mask = dense_kps_mask.reshape(num_joints, 1, output_res, output_res)
            dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
            dense_kps_mask = dense_kps_mask.reshape(num_joints * 2, output_res, output_res)
            ret.update({"dense_hps": dense_kps, "dense_hps_mask": dense_kps_mask})
            del ret["hps"], ret["hps_mask"]
        if self.opt.reg_offset:
            ret.update({"reg": reg})
        if self.opt.hm_hp:
            ret.update({"hm_hp": hm_hp})
        if self.opt.reg_hp_offset:
            ret.update({"hp_offset": hp_offset, "hp_ind": hp_ind, "hp_mask": hp_mask})
        if self.opt.debug > 0 or not self.split == "train":
            det_width = 6 + num_joints * 3
            gt_det = (
                np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1, det_width), dtype=np.float32)
            )
            meta = {"c": c, "s": s, "gt_det": gt_det, "img_id": img_id}
            ret["meta"] = meta

        if debug_aug_vis and debug_image is not None and (debug_keypoints or debug_boxes):
            overlay = debug_image.copy()
            h_vis, w_vis = overlay.shape[:2]
            summary_tokens = []
            if debug_boxes:
                for box_info in debug_boxes:
                    x1, y1, x2, y2 = box_info["bbox"]
                    pt1 = (int(round(x1)), int(round(y1)))
                    pt2 = (int(round(x2)), int(round(y2)))
                    cv2.rectangle(overlay, pt1, pt2, (255, 165, 0), 1)
                    label = "cls{}".format(box_info["cls_id"])
                    if box_info.get("cls_name"):
                        label = box_info["cls_name"]
                    text_org = (pt1[0], max(pt1[1] - 4, 0))
                    cv2.putText(
                        overlay,
                        label,
                        text_org,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (255, 165, 0),
                        1,
                        cv2.LINE_AA,
                    )
            if debug_keypoints:
                for entry in debug_keypoints:
                    if len(entry) == 3:
                        x, y, vis_flag = entry
                        kp_idx = -1
                    else:
                        x, y, vis_flag, kp_idx = entry
                    if np.isnan(x) or np.isnan(y):
                        continue
                    if w_vis == 0 or h_vis == 0:
                        continue
                    orig_cx, orig_cy = int(round(x)), int(round(y))
                    cx = min(max(orig_cx, 0), w_vis - 1)
                    cy = min(max(orig_cy, 0), h_vis - 1)
                    clipped = (orig_cx != cx) or (orig_cy != cy)
                    color = (0, 255, 0) if vis_flag > 0 else (0, 0, 255)
                    cv2.circle(overlay, (cx, cy), 3, color, -1)
                    if vis_flag <= 0:
                        cv2.line(overlay, (cx - 4, cy - 4), (cx + 4, cy + 4), color, 1)
                        cv2.line(overlay, (cx - 4, cy + 4), (cx + 4, cy - 4), color, 1)
                    label = "{}:{}".format(kp_idx if kp_idx >= 0 else "-", int(vis_flag))
                    if clipped:
                        label += "*"
                    if kp_idx >= 0:
                        summary_tokens.append(label)
                    text_org = (min(cx + 5, w_vis - 1), max(cy - 5, 0))
                    cv2.putText(
                        overlay,
                        label,
                        text_org,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

            if summary_tokens:
                summary_text = "kps " + " ".join(sorted(summary_tokens))
                cv2.putText(
                    overlay,
                    summary_text,
                    (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            save_dir = getattr(self, "_debug_aug_tmp_dir", None)
            if save_dir:
                base_name = os.path.splitext(file_name)[0]
                timestamp_str = int(item_timestamp * 1000)
                overlay_path = os.path.join(save_dir, "{}_aug_{}.jpg".format(base_name, timestamp_str))
                if cv2.imwrite(overlay_path, overlay):
                    log_msg = "[multi_pose debug] saved augmented vis: {}".format(overlay_path)
                    if sample_debug:
                        print(log_msg, flush=True)
                    try:
                        with open(os.path.join(save_dir, "aug_vis_log.txt"), "a") as log_f:
                            log_f.write(log_msg + "\n")
                    except OSError as exc:
                        if sample_debug:
                            print("[multi_pose debug] failed to append log: {}".format(exc), flush=True)
                else:
                    if sample_debug:
                        print("[multi_pose debug] failed to save augmented vis: {}".format(overlay_path), flush=True)

                def _save_heatmap(tag, heatmap_2d):
                    if heatmap_2d is None or heatmap_2d.size == 0:
                        return None
                    heatmap_norm = np.clip(heatmap_2d, 0.0, 1.0)
                    heatmap_u8 = np.round(heatmap_norm * 255.0).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
                    heatmap_color = cv2.resize(
                        heatmap_color,
                        (self.opt.input_res, self.opt.input_res),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    heatmap_path = os.path.join(save_dir, "{}_{}_{}.jpg".format(base_name, tag, timestamp_str))
                    overlay_img = cv2.addWeighted(debug_image.astype(np.uint8), 0.6, heatmap_color, 0.4, 0.0)
                    overlay_path = os.path.join(save_dir, "{}_{}_overlay_{}.jpg".format(base_name, tag, timestamp_str))
                    cv2.imwrite(heatmap_path, heatmap_color)
                    cv2.imwrite(overlay_path, overlay_img)
                    return heatmap_path, overlay_path

                hm_paths = _save_heatmap("hm", np.max(hm, axis=0) if hm.size > 0 else None)

                kp_paths = []
                if num_joints > 0:
                    for j in range(num_joints):
                        result = _save_heatmap("hm_kp{:02d}".format(j), hm_hp[j])
                        if result is not None:
                            kp_paths.append(result[0])
                    kp_all = _save_heatmap("hm_all_kps", np.max(hm_hp, axis=0))

                    if sample_debug and (hm_paths or kp_paths or kp_all):
                        print(
                            "[multi_pose debug] saved heatmaps for {} (det={}, joints={}, combined={})".format(
                                file_name, hm_paths, len(kp_paths), kp_all
                            ),
                            flush=True,
                        )
        return ret
