from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds
from .ddd_utils import ddd2locrot


def get_pred_depth(depth):
  return depth

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)
  

def ddd_post_process_2d(dets, c, s, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  include_wh = dets.shape[2] > 16
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
    classes = dets[i, :, -1]
    for j in range(opt.num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :3].astype(np.float32),
        get_alpha(dets[i, inds, 3:11])[:, np.newaxis].astype(np.float32),
        get_pred_depth(dets[i, inds, 11:12]).astype(np.float32),
        dets[i, inds, 12:15].astype(np.float32)], axis=1)
      if include_wh:
        top_preds[j + 1] = np.concatenate([
          top_preds[j + 1],
          transform_preds(
            dets[i, inds, 15:17], c[i], s[i], (opt.output_w, opt.output_h))
          .astype(np.float32)], axis=1)
    ret.append(top_preds)
  return ret

def ddd_post_process_3d(dets, calibs):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  for i in range(len(dets)):
    preds = {}
    for cls_ind in dets[i].keys():
      preds[cls_ind] = []
      for j in range(len(dets[i][cls_ind])):
        center = dets[i][cls_ind][j][:2]
        score = dets[i][cls_ind][j][2]
        alpha = dets[i][cls_ind][j][3]
        depth = dets[i][cls_ind][j][4]
        dimensions = dets[i][cls_ind][j][5:8]
        wh = dets[i][cls_ind][j][8:10]
        locations, rotation_y = ddd2locrot(
          center, alpha, dimensions, depth, calibs[0])
        bbox = [center[0] - wh[0] / 2, center[1] - wh[1] / 2,
                center[0] + wh[0] / 2, center[1] + wh[1] / 2]
        pred = [alpha] + bbox + dimensions.tolist() + \
               locations.tolist() + [rotation_y, score]
        preds[cls_ind].append(pred)
      preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)
    ret.append(preds)
  return ret

def ddd_post_process(dets, c, s, calibs, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  dets = ddd_post_process_2d(dets, c, s, opt)
  dets = ddd_post_process_3d(dets, calibs)
  return dets


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret


def multi_pose_post_process(dets, c, s, h, w, num_joints=None):
  # dets: batch x max_dets x (4 + 1 + 2 * num_joints + opt_vis + 1)
  # return list where each detection is [bbox(4), score(1), keypoints(2 * num_joints), optional visibility(num_joints)]
  ret = []
  B, max_dets, dim = dets.shape
  print(f"[post_process] Input: dets.shape={dets.shape}, num_joints={num_joints}")
  print(f"[post_process] First det before transform: bbox={dets[0,0,:4]}, score={dets[0,0,4]}, class={dets[0,0,-1]}")
  inferred_joints = num_joints
  if inferred_joints is None:
    if dim < 6:
      inferred_joints = 0
    else:
      # try to infer assuming optional visibility block may be present
      remaining = dim - 6
      if remaining <= 0:
        inferred_joints = 0
      elif remaining % 3 == 0:
        inferred_joints = remaining // 3
      elif remaining % 2 == 0:
        inferred_joints = remaining // 2
      else:
        # fallback to even part
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
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    bbox = bbox.reshape(-1, 4)
    score = dets[i, :, 4:5]

    pieces = [bbox, score]
    print(f"[post_process] After bbox transform: bbox[0]={bbox[0]}, score[0]={score[0]}")

    if inferred_joints > 0 and keypoint_end <= dim:
      pts = transform_preds(
        dets[i, :, 5:keypoint_end].reshape(-1, 2), c[i], s[i], (w, h))
      pts = pts.reshape(-1, keypoint_len)
      pieces.append(pts)
      print(f"[post_process] Added keypoints: pts[0, 0:4]={pts[0, 0:4]}")
    if vis_len > 0 and vis_end <= dim:
      pieces.append(dets[i, :, vis_start:vis_end])
      print(f"[post_process] Added visibility: vis_start={vis_start}, vis_end={vis_end}, vis[0, 0:4]={dets[i, 0, vis_start:vis_end][:4]}")

    combined = np.concatenate(pieces, axis=1).astype(np.float32)
    print(f"[post_process] After concatenate: len={combined.shape[1]}, first score={combined[0, 4] if combined.shape[0] > 0 else 'n/a'}")

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
      print(f"[post_process] class {cls_key}: kept {len(top_preds[cls_key])} entries")
    ret.append(top_preds)
  return ret
