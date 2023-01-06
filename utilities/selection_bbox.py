import numpy as np
import torch

def calc_iou_overlap(bboxes, anno_bb):
    anno_bb = np.array([anno_bb] * bboxes.shape[0])
    bboxes = torch.FloatTensor(bboxes)
    anno_bb = torch.FloatTensor(anno_bb)
    tl = torch.max(bboxes[:, :2], anno_bb[:, :2])
    br = torch.min(bboxes[:, :2] + bboxes[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
    sz = (br - tl + 1.0).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = bboxes[:, 2:].prod(dim=1) + anno_bb[:, 2:].prod(dim=1) - intersection

    return intersection / union

def selection_bbox(bboxes, target, threshold):

    iou_result = np.array(calc_iou_overlap(bboxes, target))
    iou_max = np.argmax(iou_result)
    iou_min = np.argmin(iou_result)
    # print(iou_result[iou_max])
    if iou_result[iou_max] >= threshold: ## threshold
        return bboxes[iou_max]
    else:
        return None