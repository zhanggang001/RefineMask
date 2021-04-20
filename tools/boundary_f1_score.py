import os
import cv2
import json
import pdb
import math
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm
from collections import defaultdict
from skimage.morphology import disk
from argparse import ArgumentParser
from multiprocessing.pool import Pool

from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params


class F1RunningScore(object):
    def __init__(self, num_classes=None, boundary_threshold=1, num_proc=15):

        self.n_classes = num_classes
        self.boundary_threshold = boundary_threshold
        self.num_proc = num_proc
        self.ignore_index = -1

        self.pool = Pool(processes=num_proc)
        self._Fpc = 0
        self._Fc = 0

        self.seg_map_cache = []
        self.gt_map_cache = []

    def _update_cache(self, seg_map, gt_map):
        """
        Append inputs to `seg_map_cache` and `gt_map_cache`.

        Returns whether the length reached our pool size.
        """
        self.seg_map_cache.extend(seg_map)
        self.gt_map_cache.extend(gt_map)
        return len(self.gt_map_cache) >= self.num_proc

    def _get_from_cache(self):

        n = self.num_proc
        seg_map, self.seg_map_cache = self.seg_map_cache[:n], self.seg_map_cache[n:]
        gt_map, self.gt_map_cache = self.gt_map_cache[:n], self.gt_map_cache[n:]

        return seg_map, gt_map

    def update(self, seg_map, gt_map):

        if self._update_cache(seg_map, gt_map):
            seg_map, gt_map = self._get_from_cache()
            self._update_scores(seg_map, gt_map)
        else:
            return

    def _update_scores(self, seg_map, gt_map):
        batch_size = len(seg_map)
        if batch_size == 0:
            return

        Fpc = np.zeros(self.n_classes)
        Fc = np.zeros(self.n_classes)

        for class_id in range(self.n_classes):
            args = []
            for i in range(batch_size):
                if seg_map[i].shape[0] == self.n_classes:
                    pred_i = seg_map[i][class_id] > 0.5
                    pred_is_boundary = True
                else:
                    pred_i = seg_map[i] == class_id
                    pred_is_boundary = False

                args.append([(pred_i).astype(np.uint8), (gt_map[i] == class_id).astype(np.uint8), (gt_map[i] == -1),
                             self.boundary_threshold, class_id, pred_is_boundary])
            results = self.pool.map(db_eval_boundary, args)
            results = np.array(results)
            Fs = results[:, 0]
            _valid = ~np.isnan(Fs)
            Fc[class_id] = np.sum(_valid)
            Fs[np.isnan(Fs)] = 0
            Fpc[class_id] = sum(Fs)

        self._Fc = self._Fc + Fc
        self._Fpc = self._Fpc + Fpc

    def get_scores(self):

        if self.seg_map_cache is None:
            return 0, 0

        self._update_scores(self.seg_map_cache, self.gt_map_cache)

        F_score = np.sum(self._Fpc / self._Fc) / self.n_classes
        F_score_classwise = self._Fpc / self._Fc

        return F_score, F_score_classwise

    def reset(self):
        self._Fpc = self._Fc = 0


def db_eval_boundary(args):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
            foreground_mask (ndarray): binary segmentation image.
            gt_mask         (ndarray): binary annotated image.

    Returns:
            F (float): boundaries F-measure
            P (float): boundaries precision
            R (float): boundaries recall
    """

    foreground_mask, gt_mask, ignore_mask, bound_th, class_id, pred_is_boundary = args

    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

    foreground_mask[ignore_mask] = 0
    gt_mask[ignore_mask] = 0

    # Get the pixel boundaries of both masks
    if pred_is_boundary:
        fg_boundary = foreground_mask
    else:
        fg_boundary = seg2bmap(foreground_mask)
    gt_boundary = seg2bmap(gt_mask)

    def binary_dilation(x, d):
        return cv2.dilate(x.astype(np.uint8), d).astype(np.bool)

    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F, precision


def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    Arguments:
            seg     : Segments labeled from 1..k.
            width   :   Width of desired bmap  <= seg.shape[1]
            height  :   Height of desired bmap <= seg.shape[0]

    Returns:
            bmap (ndarray): Binary boundary map.

     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) > 0.01),\
        'Can''t convert %dx%d seg to %dx%d bmap.' % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def compute_f_score(dt, gt, f1runnings):
    img_ann_map = dt.img_ann_map

    for img_id in tqdm(gt.get_img_ids()):
        instances = img_ann_map[img_id]

        ann_ids = gt.get_ann_ids(img_ids=[img_id, ])
        gt_anns = gt.load_anns(ann_ids)

        gt_img_masks = [gt.ann_to_mask(ann) for ann in gt_anns]

        for gt_ann, gt_mask in zip(gt_anns, gt_img_masks):
            dt_anns = [ins for ins in instances if ins['category_id'] == gt_ann['category_id']]
            if len(dt_anns) == 0:
                continue

            dt_masks = [gt.ann_to_rle(ann, ) for ann in dt_anns]
            ious = mask.iou(dt_masks, [gt.annToRLE(gt_ann), ], [gt_ann.get('iscrowd', 0), ]).reshape(-1)

            if ious.max() < 0.5:
                continue

            x, y, w, h = map(int, gt_ann['bbox'])
            w, h = max(1, w), max(1, h)

            max_dt_ann = dt_anns[ious.argmax()]
            dt_mask = mask.decode(max_dt_ann['segmentation'])

            dt_mask = dt_mask[max(y - h // 2, 0): y + h + h // 2, max(x - w // 2, 0): x + w + w // 2]
            gt_mask = gt_mask[max(y - h // 2, 0): y + h + h // 2, max(x - w // 2, 0): x + w + w // 2]

            if gt_mask.shape[0] <= 1 or gt_mask.shape[1] <= 1:
                continue

            for f1running in f1runnings:
                f1running.update(dt_mask[np.newaxis], gt_mask[np.newaxis])


def run_f_score(gt_file, pd_file):
    f1running_1 = F1RunningScore(num_classes=1, boundary_threshold=1)
    f1running_3 = F1RunningScore(num_classes=1, boundary_threshold=3)
    f1running_5 = F1RunningScore(num_classes=1, boundary_threshold=5)

    print('load gt json')
    coco_gt = COCO(gt_file)

    print('load pred json')
    coco_dt = coco_gt.loadRes(pd_file)

    print('compute f score')
    compute_f_score(coco_dt, coco_gt, [f1running_1, f1running_3, f1running_5])

    print("F1 boundary score (1): ", f1running_1.get_scores()[0])
    print("F1 boundary score (3): ", f1running_3.get_scores()[0])
    print("F1 boundary score (5): ", f1running_5.get_scores()[0])


if __name__ == "__main__":
    run_f_score('jsons/lvis_v0.5_val_cocofied.json', 'jsons/coco_maskrcnn.segm.json')
    run_f_score('jsons/lvis_v0.5_val_cocofied.json', 'jsons/coco_refinemask.segm.json')
