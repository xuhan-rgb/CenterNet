from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
import torch.nn as nn
from .utils import _gather_feat, _transpose_and_gather_feat

# Lightweight colored logger (ANSI). Toggle with env DEBUG_DECODE=1
_USE_COLOR = True
_DEBUG_DECODE = os.environ.get("DEBUG_DECODE", "0") == "1"


def _c(code, text):
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def log_info(msg):
    print(_c("32", f"[INFO] {msg}"))  # green


def log_warn(msg):
    print(_c("33", f"[WARN] {msg}"))  # yellow


def log_error(msg):
    print(_c("31", f"[ERROR] {msg}"))  # red


def log_debug(msg):
    if _DEBUG_DECODE:
        print(_c("36", f"[DEBUG] {msg}"))  # cyan


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _left_aggregate(heat):
    """
    heat: batchsize x channels x h x w
    """
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = heat[i] >= heat[i - 1]
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)


def _right_aggregate(heat):
    """
    heat: batchsize x channels x h x w
    """
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = heat[i] >= heat[i + 1]
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)


def _top_aggregate(heat):
    """
    heat: batchsize x channels x h x w
    """
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = heat[i] >= heat[i - 1]
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)


def _bottom_aggregate(heat):
    """
    heat: batchsize x channels x h x w
    """
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = heat[i] >= heat[i + 1]
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)


def _h_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _left_aggregate(heat) + aggr_weight * _right_aggregate(heat) + heat


def _v_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _top_aggregate(heat) + aggr_weight * _bottom_aggregate(heat) + heat


"""
# Slow for large number of categories
def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
"""


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def agnex_ct_decode(
    t_heat,
    l_heat,
    b_heat,
    r_heat,
    ct_heat,
    t_regr=None,
    l_regr=None,
    b_regr=None,
    r_regr=None,
    K=40,
    scores_thresh=0.1,
    center_thresh=0.1,
    aggr_weight=0.0,
    num_dets=1000,
):
    batch, cat, height, width = t_heat.size()

    """
    t_heat  = torch.sigmoid(t_heat)
    l_heat  = torch.sigmoid(l_heat)
    b_heat  = torch.sigmoid(b_heat)
    r_heat  = torch.sigmoid(r_heat)
    ct_heat = torch.sigmoid(ct_heat)
    """
    if aggr_weight > 0:
        t_heat = _h_aggregate(t_heat, aggr_weight=aggr_weight)
        l_heat = _v_aggregate(l_heat, aggr_weight=aggr_weight)
        b_heat = _h_aggregate(b_heat, aggr_weight=aggr_weight)
        r_heat = _v_aggregate(r_heat, aggr_weight=aggr_weight)

    # perform nms on heatmaps
    t_heat = _nms(t_heat)
    l_heat = _nms(l_heat)
    b_heat = _nms(b_heat)
    r_heat = _nms(r_heat)

    t_heat[t_heat > 1] = 1
    l_heat[l_heat > 1] = 1
    b_heat[b_heat > 1] = 1
    r_heat[r_heat > 1] = 1

    t_scores, t_inds, _, t_ys, t_xs = _topk(t_heat, K=K)
    l_scores, l_inds, _, l_ys, l_xs = _topk(l_heat, K=K)
    b_scores, b_inds, _, b_ys, b_xs = _topk(b_heat, K=K)
    r_scores, r_inds, _, r_ys, r_xs = _topk(r_heat, K=K)

    ct_heat_agn, ct_clses = torch.max(ct_heat, dim=1, keepdim=True)

    # import pdb; pdb.set_trace()

    t_ys = t_ys.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    t_xs = t_xs.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_ys = l_ys.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    l_xs = l_xs.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_ys = b_ys.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    b_xs = b_xs.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_ys = r_ys.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    r_xs = r_xs.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)

    box_ct_xs = ((l_xs + r_xs + 0.5) / 2).long()
    box_ct_ys = ((t_ys + b_ys + 0.5) / 2).long()

    ct_inds = box_ct_ys * width + box_ct_xs
    ct_inds = ct_inds.view(batch, -1)
    ct_heat_agn = ct_heat_agn.view(batch, -1, 1)
    ct_clses = ct_clses.view(batch, -1, 1)
    ct_scores = _gather_feat(ct_heat_agn, ct_inds)
    clses = _gather_feat(ct_clses, ct_inds)

    t_scores = t_scores.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_scores = l_scores.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_scores = b_scores.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_scores = r_scores.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    ct_scores = ct_scores.view(batch, K, K, K, K)
    scores = (t_scores + l_scores + b_scores + r_scores + 2 * ct_scores) / 6

    # reject boxes based on classes
    top_inds = (t_ys > l_ys) + (t_ys > b_ys) + (t_ys > r_ys)
    top_inds = top_inds > 0
    left_inds = (l_xs > t_xs) + (l_xs > b_xs) + (l_xs > r_xs)
    left_inds = left_inds > 0
    bottom_inds = (b_ys < t_ys) + (b_ys < l_ys) + (b_ys < r_ys)
    bottom_inds = bottom_inds > 0
    right_inds = (r_xs < t_xs) + (r_xs < l_xs) + (r_xs < b_xs)
    right_inds = right_inds > 0

    sc_inds = (
        (t_scores < scores_thresh)
        + (l_scores < scores_thresh)
        + (b_scores < scores_thresh)
        + (r_scores < scores_thresh)
        + (ct_scores < center_thresh)
    )
    sc_inds = sc_inds > 0

    scores = scores - sc_inds.float()
    scores = scores - top_inds.float()
    scores = scores - left_inds.float()
    scores = scores - bottom_inds.float()
    scores = scores - right_inds.float()

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    if t_regr is not None and l_regr is not None and b_regr is not None and r_regr is not None:
        t_regr = _transpose_and_gather_feat(t_regr, t_inds)
        t_regr = t_regr.view(batch, K, 1, 1, 1, 2)
        l_regr = _transpose_and_gather_feat(l_regr, l_inds)
        l_regr = l_regr.view(batch, 1, K, 1, 1, 2)
        b_regr = _transpose_and_gather_feat(b_regr, b_inds)
        b_regr = b_regr.view(batch, 1, 1, K, 1, 2)
        r_regr = _transpose_and_gather_feat(r_regr, r_inds)
        r_regr = r_regr.view(batch, 1, 1, 1, K, 2)

        t_xs = t_xs + t_regr[..., 0]
        t_ys = t_ys + t_regr[..., 1]
        l_xs = l_xs + l_regr[..., 0]
        l_ys = l_ys + l_regr[..., 1]
        b_xs = b_xs + b_regr[..., 0]
        b_ys = b_ys + b_regr[..., 1]
        r_xs = r_xs + r_regr[..., 0]
        r_ys = r_ys + r_regr[..., 1]
    else:
        t_xs = t_xs + 0.5
        t_ys = t_ys + 0.5
        l_xs = l_xs + 0.5
        l_ys = l_ys + 0.5
        b_xs = b_xs + 0.5
        b_ys = b_ys + 0.5
        r_xs = r_xs + 0.5
        r_ys = r_ys + 0.5

    bboxes = torch.stack((l_xs, t_ys, r_xs, b_ys), dim=5)
    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses = clses.contiguous().view(batch, -1, 1)
    clses = _gather_feat(clses, inds).float()

    t_xs = t_xs.contiguous().view(batch, -1, 1)
    t_xs = _gather_feat(t_xs, inds).float()
    t_ys = t_ys.contiguous().view(batch, -1, 1)
    t_ys = _gather_feat(t_ys, inds).float()
    l_xs = l_xs.contiguous().view(batch, -1, 1)
    l_xs = _gather_feat(l_xs, inds).float()
    l_ys = l_ys.contiguous().view(batch, -1, 1)
    l_ys = _gather_feat(l_ys, inds).float()
    b_xs = b_xs.contiguous().view(batch, -1, 1)
    b_xs = _gather_feat(b_xs, inds).float()
    b_ys = b_ys.contiguous().view(batch, -1, 1)
    b_ys = _gather_feat(b_ys, inds).float()
    r_xs = r_xs.contiguous().view(batch, -1, 1)
    r_xs = _gather_feat(r_xs, inds).float()
    r_ys = r_ys.contiguous().view(batch, -1, 1)
    r_ys = _gather_feat(r_ys, inds).float()

    detections = torch.cat([bboxes, scores, t_xs, t_ys, l_xs, l_ys, b_xs, b_ys, r_xs, r_ys, clses], dim=2)

    return detections


def exct_decode(
    t_heat,
    l_heat,
    b_heat,
    r_heat,
    ct_heat,
    t_regr=None,
    l_regr=None,
    b_regr=None,
    r_regr=None,
    K=40,
    scores_thresh=0.1,
    center_thresh=0.1,
    aggr_weight=0.0,
    num_dets=1000,
):
    batch, cat, height, width = t_heat.size()
    """
    t_heat  = torch.sigmoid(t_heat)
    l_heat  = torch.sigmoid(l_heat)
    b_heat  = torch.sigmoid(b_heat)
    r_heat  = torch.sigmoid(r_heat)
    ct_heat = torch.sigmoid(ct_heat)
    """

    if aggr_weight > 0:
        t_heat = _h_aggregate(t_heat, aggr_weight=aggr_weight)
        l_heat = _v_aggregate(l_heat, aggr_weight=aggr_weight)
        b_heat = _h_aggregate(b_heat, aggr_weight=aggr_weight)
        r_heat = _v_aggregate(r_heat, aggr_weight=aggr_weight)

    # perform nms on heatmaps
    t_heat = _nms(t_heat)
    l_heat = _nms(l_heat)
    b_heat = _nms(b_heat)
    r_heat = _nms(r_heat)

    t_heat[t_heat > 1] = 1
    l_heat[l_heat > 1] = 1
    b_heat[b_heat > 1] = 1
    r_heat[r_heat > 1] = 1

    t_scores, t_inds, t_clses, t_ys, t_xs = _topk(t_heat, K=K)
    l_scores, l_inds, l_clses, l_ys, l_xs = _topk(l_heat, K=K)
    b_scores, b_inds, b_clses, b_ys, b_xs = _topk(b_heat, K=K)
    r_scores, r_inds, r_clses, r_ys, r_xs = _topk(r_heat, K=K)

    t_ys = t_ys.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    t_xs = t_xs.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_ys = l_ys.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    l_xs = l_xs.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_ys = b_ys.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    b_xs = b_xs.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_ys = r_ys.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    r_xs = r_xs.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)

    t_clses = t_clses.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_clses = l_clses.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_clses = b_clses.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_clses = r_clses.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    box_ct_xs = ((l_xs + r_xs + 0.5) / 2).long()
    box_ct_ys = ((t_ys + b_ys + 0.5) / 2).long()
    ct_inds = t_clses.long() * (height * width) + box_ct_ys * width + box_ct_xs
    ct_inds = ct_inds.view(batch, -1)
    ct_heat = ct_heat.view(batch, -1, 1)
    ct_scores = _gather_feat(ct_heat, ct_inds)

    t_scores = t_scores.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_scores = l_scores.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_scores = b_scores.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_scores = r_scores.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    ct_scores = ct_scores.view(batch, K, K, K, K)
    scores = (t_scores + l_scores + b_scores + r_scores + 2 * ct_scores) / 6

    # reject boxes based on classes
    cls_inds = (t_clses != l_clses) + (t_clses != b_clses) + (t_clses != r_clses)
    cls_inds = cls_inds > 0

    top_inds = (t_ys > l_ys) + (t_ys > b_ys) + (t_ys > r_ys)
    top_inds = top_inds > 0
    left_inds = (l_xs > t_xs) + (l_xs > b_xs) + (l_xs > r_xs)
    left_inds = left_inds > 0
    bottom_inds = (b_ys < t_ys) + (b_ys < l_ys) + (b_ys < r_ys)
    bottom_inds = bottom_inds > 0
    right_inds = (r_xs < t_xs) + (r_xs < l_xs) + (r_xs < b_xs)
    right_inds = right_inds > 0

    sc_inds = (
        (t_scores < scores_thresh)
        + (l_scores < scores_thresh)
        + (b_scores < scores_thresh)
        + (r_scores < scores_thresh)
        + (ct_scores < center_thresh)
    )
    sc_inds = sc_inds > 0

    scores = scores - sc_inds.float()
    scores = scores - cls_inds.float()
    scores = scores - top_inds.float()
    scores = scores - left_inds.float()
    scores = scores - bottom_inds.float()
    scores = scores - right_inds.float()

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    if t_regr is not None and l_regr is not None and b_regr is not None and r_regr is not None:
        t_regr = _transpose_and_gather_feat(t_regr, t_inds)
        t_regr = t_regr.view(batch, K, 1, 1, 1, 2)
        l_regr = _transpose_and_gather_feat(l_regr, l_inds)
        l_regr = l_regr.view(batch, 1, K, 1, 1, 2)
        b_regr = _transpose_and_gather_feat(b_regr, b_inds)
        b_regr = b_regr.view(batch, 1, 1, K, 1, 2)
        r_regr = _transpose_and_gather_feat(r_regr, r_inds)
        r_regr = r_regr.view(batch, 1, 1, 1, K, 2)

        t_xs = t_xs + t_regr[..., 0]
        t_ys = t_ys + t_regr[..., 1]
        l_xs = l_xs + l_regr[..., 0]
        l_ys = l_ys + l_regr[..., 1]
        b_xs = b_xs + b_regr[..., 0]
        b_ys = b_ys + b_regr[..., 1]
        r_xs = r_xs + r_regr[..., 0]
        r_ys = r_ys + r_regr[..., 1]
    else:
        t_xs = t_xs + 0.5
        t_ys = t_ys + 0.5
        l_xs = l_xs + 0.5
        l_ys = l_ys + 0.5
        b_xs = b_xs + 0.5
        b_ys = b_ys + 0.5
        r_xs = r_xs + 0.5
        r_ys = r_ys + 0.5

    bboxes = torch.stack((l_xs, t_ys, r_xs, b_ys), dim=5)
    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses = t_clses.contiguous().view(batch, -1, 1)
    clses = _gather_feat(clses, inds).float()

    t_xs = t_xs.contiguous().view(batch, -1, 1)
    t_xs = _gather_feat(t_xs, inds).float()
    t_ys = t_ys.contiguous().view(batch, -1, 1)
    t_ys = _gather_feat(t_ys, inds).float()
    l_xs = l_xs.contiguous().view(batch, -1, 1)
    l_xs = _gather_feat(l_xs, inds).float()
    l_ys = l_ys.contiguous().view(batch, -1, 1)
    l_ys = _gather_feat(l_ys, inds).float()
    b_xs = b_xs.contiguous().view(batch, -1, 1)
    b_xs = _gather_feat(b_xs, inds).float()
    b_ys = b_ys.contiguous().view(batch, -1, 1)
    b_ys = _gather_feat(b_ys, inds).float()
    r_xs = r_xs.contiguous().view(batch, -1, 1)
    r_xs = _gather_feat(r_xs, inds).float()
    r_ys = r_ys.contiguous().view(batch, -1, 1)
    r_ys = _gather_feat(r_ys, inds).float()

    detections = torch.cat([bboxes, scores, t_xs, t_ys, l_xs, l_ys, b_xs, b_ys, r_xs, r_ys, clses], dim=2)

    return detections


def ddd_decode(heat, rot, depth, dim, wh=None, reg=None, K=40):
    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5

    rot = _transpose_and_gather_feat(rot, inds)
    rot = rot.view(batch, K, 8)
    depth = _transpose_and_gather_feat(depth, inds)
    depth = depth.view(batch, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch, K, 3)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    xs = xs.view(batch, K, 1)
    ys = ys.view(batch, K, 1)

    if wh is not None:
        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)
        detections = torch.cat([xs, ys, scores, rot, depth, dim, wh, clses], dim=2)
    else:
        detections = torch.cat([xs, ys, scores, rot, depth, dim, clses], dim=2)

    return detections


def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat(
        [xs - wh[..., 0:1] / 2, ys - wh[..., 1:2] / 2, xs + wh[..., 0:1] / 2, ys + wh[..., 1:2] / 2], dim=2
    )
    detections = torch.cat([bboxes, scores, clses], dim=2)

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
    log=False,
    return_initial_kps=False,
):
    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    kps = _transpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)
    kps_before_offset = kps.clone()
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)

    # 保存bbox中心的整数坐标（应用reg offset之前）
    xs_int = xs.clone()
    ys_int = ys.clone()

    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        reg_offset = reg.clone()  # 保存reg offset用于log
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        reg_offset = None
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5

    xs_int_clone = xs_int.clone()
    ys_int_clone = ys_int.clone()
    xs_int_clone = xs_int_clone.view(batch, K, 1)
    ys_int_clone = ys_int_clone.view(batch, K, 1)

    initial_kps = kps_before_offset.view(batch, K, num_joints, 2)
    # initial_kps[..., 0] += xs
    # initial_kps[..., 1] += ys
    initial_kps[..., 0] += xs_int_clone  # * 解决bug,应该以box int中心坐标计算偏差
    initial_kps[..., 1] += ys_int_clone
    initial_kps_flat = initial_kps.view(batch, K, num_joints * 2)
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    hp_vis_scores = None
    if hp_vis is not None and num_joints > 0:
        hp_vis_pred = _transpose_and_gather_feat(hp_vis, inds)
        hp_vis_pred = hp_vis_pred.view(batch, K, num_joints)
        hp_vis_scores = torch.sigmoid(hp_vis_pred)

    bboxes = torch.cat(
        [xs - wh[..., 0:1] / 2, ys - wh[..., 1:2] / 2, xs + wh[..., 0:1] / 2, ys + wh[..., 1:2] / 2], dim=2
    )
    if hm_hp is not None:
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.view(batch, K, num_joints, 2).permute(0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K

        # 保存hm中心坐标（应用offset之前）
        hm_xs_center = hm_xs.clone()
        hm_ys_center = hm_ys.clone()

        if hp_offset is not None:
            hp_offset = _transpose_and_gather_feat(hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_joints, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        if log:
            with torch.no_grad():
                sample_joint = min(num_joints, 4)
                sample_k = min(K, 2)

                # 提取所有数据
                scores_np = hm_score[0, :sample_joint, :sample_k].cpu().numpy()
                # 应用sigmoid变换
                scores_sigmoid = 1.0 / (1.0 + np.exp(-scores_np))
                xs_np = hm_xs[0, :sample_joint, :sample_k].cpu().numpy()
                ys_np = hm_ys[0, :sample_joint, :sample_k].cpu().numpy()
                hm_xs_int = np.floor(xs_np).astype(int)
                hm_ys_int = np.floor(ys_np).astype(int)

                # 提取hm中心坐标
                xs_center_np = hm_xs_center[0, :sample_joint, :sample_k].cpu().numpy()
                ys_center_np = hm_ys_center[0, :sample_joint, :sample_k].cpu().numpy()

                if hp_offset is not None:
                    hp_offset_np = hp_offset[0, :sample_joint, :sample_k, :].cpu().numpy()
                    log_info(
                        "multi_pose: keypoint heatmap candidates (joint, cand, hm_center, offset, final_pos, score):"
                    )
                    for j in range(sample_joint):
                        for idx in range(sample_k):
                            print(
                                f"  joint {j} cand {idx}: hm_center=({xs_center_np[j, idx]:.1f}, {ys_center_np[j, idx]:.1f}), offset=({hp_offset_np[j, idx, 0]:+.4f}, {hp_offset_np[j, idx, 1]:+.4f}), final=({xs_np[j, idx]:.1f}, {ys_np[j, idx]:.1f}), score={scores_sigmoid[j, idx]:.4f}"
                            )
                else:
                    log_info("multi_pose: keypoint heatmap candidates (joint, cand, hm_center, final_pos, score):")
                    for j in range(sample_joint):
                        for idx in range(sample_k):
                            print(
                                f"  joint {j} cand {idx}: hm_center=({xs_center_np[j, idx]:.1f}, {ys_center_np[j, idx]:.1f}), final=({xs_np[j, idx]:.1f}, {ys_np[j, idx]:.1f}), score={scores_sigmoid[j, idx]:.4f}"
                            )

        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(2).expand(batch, num_joints, K, K, 2)
        dist = ((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        selected_idx = min_ind.clone()
        hm_score = hm_score.gather(2, selected_idx).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = selected_idx.view(batch, num_joints, K, 1, 1).expand(batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)
        if log:
            with torch.no_grad():
                sample_det = min(K, 1)
                sample_joint = min(num_joints, 4)
                log_info("multi_pose: selected hm candidates (det, joint, idx, hm_center, offset, final, score):")
                for det_idx in range(sample_det):
                    for joint_idx in range(sample_joint):
                        idx_val = int(selected_idx[0, joint_idx, det_idx].item())
                        hm_cx = float(hm_xs_center[0, joint_idx, idx_val].item())
                        hm_cy = float(hm_ys_center[0, joint_idx, idx_val].item())
                        final_x = float(hm_kps[0, joint_idx, det_idx, 0])
                        final_y = float(hm_kps[0, joint_idx, det_idx, 1])
                        score_raw = float(hm_score[0, joint_idx, det_idx, 0])
                        score_sigmoid = 1.0 / (1.0 + np.exp(-score_raw))
                        offset_x = final_x - hm_cx
                        offset_y = final_y - hm_cy
                        print(
                            f"  det {det_idx} joint {joint_idx}: idx={idx_val}, hm_center=({hm_cx:.2f}, {hm_cy:.2f}), "
                            f"offset=({offset_x:+.4f}, {offset_y:+.4f}), final=({final_x:.2f}, {final_y:.2f}), "
                            f"score={score_sigmoid:.4f}"
                        )
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        tb_margin_thresh = (b - t) / 4.0
        lr_margin_thresh = (r - l) / 4.0
        size_term = torch.max(b - t, r - l) * 0.3

        cond_left = hm_kps[..., 0:1] < l - lr_margin_thresh
        cond_right = hm_kps[..., 0:1] > r + lr_margin_thresh
        cond_top = hm_kps[..., 1:2] < t - tb_margin_thresh
        cond_bottom = hm_kps[..., 1:2] > b + tb_margin_thresh
        cond_score = hm_score < thresh
        cond_dist = min_dist > size_term

        mask = cond_left + cond_right + cond_top + cond_bottom + cond_score + cond_dist
        mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        if log:
            with torch.no_grad():
                sample_det = min(K, 1)
                sample_joint = min(num_joints, 4)
                log_info("multi_pose: reject checks (det, joint, reasons):")
                for det_idx in range(sample_det):
                    for joint_idx in range(sample_joint):
                        reasons = []
                        hm_score_raw = float(hm_score[0, joint_idx, det_idx, 0])
                        hm_score_sig = 1.0 / (1.0 + np.exp(-hm_score_raw))
                        dist_val = float(min_dist[0, joint_idx, det_idx, 0])
                        size_term_val = float(size_term[0, joint_idx, det_idx, 0])
                        if bool(cond_score[0, joint_idx, det_idx, 0]):
                            reasons.append(f"score<{thresh:.2f} (raw={hm_score_raw:.3f}, sig={hm_score_sig:.4f})")
                        if bool(cond_dist[0, joint_idx, det_idx, 0]):
                            reasons.append(f"dist>{size_term_val:.2f} (dist={dist_val:.2f})")
                        if bool(cond_left[0, joint_idx, det_idx, 0]):
                            reasons.append("left_margin")
                        if bool(cond_right[0, joint_idx, det_idx, 0]):
                            reasons.append("right_margin")
                        if bool(cond_top[0, joint_idx, det_idx, 0]):
                            reasons.append("top_margin")
                        if bool(cond_bottom[0, joint_idx, det_idx, 0]):
                            reasons.append("bottom_margin")
                        if not reasons:
                            reasons.append("heatmap_ok")
                        print(f"  det {det_idx} joint {joint_idx}: {', '.join(reasons)}")

        mixed_kps = (1 - mask) * hm_kps + mask * kps
        if log:
            with torch.no_grad():
                sample_det = min(K, 1)
                sample_joint = min(num_joints, 4)
                log_info("multi_pose: final keypoint source (det, joint, source, final, hm, reg, score):")
                for det_idx in range(sample_det):
                    for joint_idx in range(sample_joint):
                        mask_val = float(mask[0, joint_idx, det_idx, 0])
                        source = "heatmap" if mask_val < 0.5 else "reg"
                        final_x = float(mixed_kps[0, joint_idx, det_idx, 0])
                        final_y = float(mixed_kps[0, joint_idx, det_idx, 1])
                        hm_x = float(hm_kps[0, joint_idx, det_idx, 0])
                        hm_y = float(hm_kps[0, joint_idx, det_idx, 1])
                        reg_x = float(kps[0, joint_idx, det_idx, 0])
                        reg_y = float(kps[0, joint_idx, det_idx, 1])
                        score_raw = float(hm_score[0, joint_idx, det_idx, 0])
                        score_sigmoid = 1.0 / (1.0 + np.exp(-score_raw))
                        print(
                            f"  det {det_idx} joint {joint_idx}: source={source}, final=({final_x:.2f}, {final_y:.2f}), "
                            f"hm=({hm_x:.2f}, {hm_y:.2f}), reg=({reg_x:.2f}, {reg_y:.2f}), score={score_sigmoid:.4f}"
                        )
        kps = mixed_kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(batch, K, num_joints * 2)

    if log:
        # Note on detections tensor layout when concatenated later in this function:
        # detections = [x1, y1, x2, y2, score, kps(2*num_joints), hp_vis(num_joints, optional), cls]
        # This clarifies shapes like (1, 100, 18) seen in logs when num_joints == 4 and hp_vis is enabled.
        with torch.no_grad():
            sample = min(K, 1)  # keep log concise: show only the first detection by default
            init_np = initial_kps[:1, :sample, :, :].cpu().numpy()
            # compute bbox center and per-keypoint offsets for the first sample
            b_np = bboxes[:1, :sample, :].cpu().numpy()  # shape (1, sample, 4)
            cx = (b_np[..., 0] + b_np[..., 2]) / 2.0
            cy = (b_np[..., 1] + b_np[..., 3]) / 2.0
            # offsets = kps - center
            off_np = init_np.copy()
            off_np[..., 0] = off_np[..., 0] - cx[..., None]
            off_np[..., 1] = off_np[..., 1] - cy[..., None]

            log_info("decode: bbox center and keypoint offsets:")
            # 提取xs_int, ys_int, scores和reg_offset用于打印
            # xs_int和ys_int是1维tensor，但xs.view会给它加一维，所以需要squeeze或者flatten
            xs_int_np = xs_int.cpu().numpy().flatten()
            ys_int_np = ys_int.cpu().numpy().flatten()
            scores_np = scores[:sample, :].cpu().numpy()
            # 应用sigmoid到scores
            scores_sigmoid = 1.0 / (1.0 + np.exp(-scores_np))

            for s in range(sample):
                # 打印bbox中心信息
                cx_int_val = int(xs_int_np[s])
                cy_int_val = int(ys_int_np[s])
                score_val = float(scores_sigmoid[0, s, 0])
                if reg_offset is not None:
                    reg_np = reg_offset[:sample, :].cpu().numpy()
                    # reg_offset的shape是(batch, K, 2)，所以需要索引[0, s, :]
                    reg_x = float(reg_np[0, s, 0])
                    reg_y = float(reg_np[0, s, 1])
                    print(
                        f"  detection {s}: center_int=({cx_int_val}, {cy_int_val}), reg_offset=({reg_x:+.4f}, {reg_y:+.4f}), center_final=({cx[0, s]:.2f}, {cy[0, s]:.2f}), score={score_val:.4f}"
                    )
                else:
                    print(
                        f"  detection {s}: center_int=({cx_int_val}, {cy_int_val}), center_final=({cx[0, s]:.2f}, {cy[0, s]:.2f}), score={score_val:.4f}"
                    )

                for j in range(num_joints):
                    init_x = init_np[0, s, j, 0]
                    init_y = init_np[0, s, j, 1]
                    off_x = off_np[0, s, j, 0]
                    off_y = off_np[0, s, j, 1]
                    print(f"    joint {j}: initial=({init_x:.2f}, {init_y:.2f}), offset=({off_x:+.2f}, {off_y:+.2f})")

            final_np = kps[:1, :sample, :].cpu().numpy().reshape(1, sample, num_joints, 2)
            log_info("decode: initial kps (from offset) first {}:".format(sample))
            print(init_np)
            log_info("decode: final kps (after hm refine) first {}:".format(sample))
            print(final_np)

            # 计算并显示diff
            diff_np = final_np - init_np
            diff_dist = np.sqrt(diff_np[..., 0] ** 2 + diff_np[..., 1] ** 2)
            log_info("decode: keypoint corrections (final - initial):")
            for s in range(sample):
                for j in range(num_joints):
                    dx = diff_np[0, s, j, 0]
                    dy = diff_np[0, s, j, 1]
                    dist = diff_dist[0, s, j]
                    status = "MODIFIED" if dist > 0.5 else "unchanged"
                    print(f"  joint {j}: dx={dx:+.2f}, dy={dy:+.2f}, dist={dist:.2f} [{status}]")

    pieces = [bboxes, scores, kps]
    if hp_vis_scores is not None:
        if log:
            with torch.no_grad():
                sample = min(K, 1)
                sample_joints = min(num_joints, 4)
                vis_np = hp_vis_scores[:1, :sample, :sample_joints].cpu().numpy()
                log_info(
                    "multi_pose: predicted visibility (first det x first joints): "
                    + " ".join(
                        f"j{j}:{vis_np[0, 0, j]:.3f}"
                        for j in range(sample_joints)
                    )
                )
        pieces.append(hp_vis_scores)
    pieces.append(clses)
    detections = torch.cat(pieces, dim=2)

    if return_initial_kps:
        return detections, initial_kps_flat
    return detections
