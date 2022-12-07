# Online System
import os, sys, cv2, torch
import multiprocessing
import numpy as np
import importlib
import argparse
from utilities.load_yaml import get_params

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
init, update = False, False
cfg = get_params('./config.yaml')
tracker_cfg = get_parameters(cfg.tracker, cfg.tracker_params)

## Initialize
tracker = OSTrack(tracker_cfg, None, threshold=1.0)

for i in range(1):

    ## Detection
    if not init:
        bboxes = detect(cfg)
        if bboxes == []:
            continue
    elif update:
        bboxes = detect(cfg)

    ## Selective Module
    # bbox = {'init_bbox': [691, 365, 36, 22]}
    bboxes = np.array(bboxes)
    bboxes[:, 2] -= bboxes[:, 0]
    bboxes[:, 3] -= bboxes[:, 1]
    bbox = {'init_bbox': bboxes[0]}

    ## Tracker
    img = cv2.imread('./test/000001.jpg')
    img2 = cv2.imread('./test/000002.jpg')
    
    if (not init) or update:
        init, update = True, False
        tracker.initialize(img, bbox)
    output = tracker.track(img2)
    update = output['update_template']
    print(output)

    ## Show Results
    box = output['target_bbox']
    box = [int(float(i)) for i in box]
    color = (255, 0, 0)
    image = cv2.rectangle(img2, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, thickness = 2)
    cv2.imwrite('test.png', image)