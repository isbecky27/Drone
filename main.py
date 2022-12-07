# Online System
import os, sys, cv2, torch
import multiprocessing
import numpy as np
import importlib
import argparse
from utilities.load_yaml import get_params
from utilities.selection_bbox import selection_bbox

## Yolov7
prj_path = os.path.join(os.path.dirname(__file__), './yolov7')
if prj_path not in sys.path:
    sys.path.append(prj_path)
from detect import detect

## OSTrack
prj_path = os.path.join(os.path.dirname(__file__), './OSTrack')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation.environment import env_settings
from lib.test.evaluation import get_dataset
from lib.test.tracker import OSTrack

def get_parameters(tracker, tracker_params):
    """Get parameters."""
    param_module = importlib.import_module('lib.test.parameter.{}'.format(tracker))
    params = param_module.parameters(tracker_params)
    params.debug = False
    return params

## Get Parameters
cfg = get_params('./config.yaml')
tracker_cfg = get_parameters(cfg.tracker, cfg.tracker_params)

## Initialize
tracker = OSTrack(tracker_cfg, None, threshold=2.0)

seq = 'person20'
prev_bbox = None
init, update = False, False
for idx in range(1, len(os.listdir(f'./data/{seq}/'))+1):

    ## Detection
    if not init:
        cfg.source = f'./data/{seq}/{str(idx).zfill(6)}.jpg'
        bboxes = detect(cfg)
        if bboxes == []:
            continue
    elif update:
        cfg.source = f'./data/{seq}/{str(idx).zfill(6)}.jpg'
        bboxes = detect(cfg)
        print("Detection:", idx)

    ## Manual Selection
    if not init:
        bbox = {'init_bbox': [453,367,69,159]}
        prev_bbox = bbox['init_bbox']
    ## Selection Module
    else:
        bboxes = np.array(bboxes)
        bboxes[:, 2] -= bboxes[:, 0]
        bboxes[:, 3] -= bboxes[:, 1]
        target = selection_bbox(bboxes, prev_bbox)
        bbox = {'init_bbox': target}

    ## Tracker
    img = cv2.imread(f'./data/{seq}/{str(idx).zfill(6)}.jpg')
    if (not init) or (update and (target is not None)):
        init, update = True, False
        tracker.initialize(img, bbox)
    output = tracker.track(img)
    update = output['update_template']
    if update:
        print("update:", idx)
    else:
        prev_bbox = output['target_bbox']

    ## Show Results
    result_path = f'./output/{seq}'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    box = output['target_bbox']
    box = [int(float(i)) for i in box]
    color = (255, 0, 0)
    image = cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, thickness = 2)
    cv2.imwrite(f'{result_path}/{str(idx).zfill(6)}.jpg', image)