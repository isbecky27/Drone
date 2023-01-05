# Online System
import os, sys, cv2, torch
import multiprocessing
import numpy as np
import importlib
import argparse
from utilities.load_yaml import get_params
from utilities.selection_bbox import selection_bbox
from utilities.manual_selection import manual_selection

## Yolov7
prj_path = os.path.join(os.path.dirname(__file__), './yolov7')
if prj_path not in sys.path:
    sys.path.append(prj_path)
from detect import init_model, detect_simple
from utils.general import increment_path

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

## LoadImages
from utilities.datasets import LoadImages
dataset = LoadImages(cfg.source)

## Directories
from pathlib import Path
save_dir = Path(increment_path(Path(cfg.project) / cfg.name, exist_ok=cfg.exist_ok))  # increment run
(save_dir / 'labels' if cfg.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
cfg.save_dir = save_dir

## Initialize
tracker = OSTrack(tracker_cfg, None, threshold=1.0) # OSTrack
init_model(cfg) # Yolov7

prev_bbox = None
init, update = False, False
for idx, (path, img) in enumerate(dataset, start=1):

    ## Detection
    if not init:
        result_img, bboxes = detect_simple(cfg, path, img)
        if bboxes == []:
            continue
    elif update:
        result_img, bboxes = detect_simple(cfg, path, img)
        print("Detection:", idx)

    ## Manual Selection
    if not init:
        selected_bbox = manual_selection(result_img, bboxes)
        # print(selected_bbox)
        if selected_bbox is not None:
            selected_bbox[2] -= selected_bbox[0]
            selected_bbox[3] -= selected_bbox[1]
            bbox = {'init_bbox': selected_bbox}
            prev_bbox = bbox['init_bbox']
        else:
            continue
    ## Selection Module
    else:
        bboxes = np.array(bboxes)
        bboxes[:, 2] -= bboxes[:, 0]
        bboxes[:, 3] -= bboxes[:, 1]
        target = selection_bbox(bboxes, prev_bbox)
        bbox = {'init_bbox': target}

    ## Tracker
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
    box = output['target_bbox']
    box = [int(float(i)) for i in box]
    color = (255, 0, 0)
    image = cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, thickness = 2)
    p = Path(path)  # to Path
    save_path = str(save_dir / p.name)
    cv2.imwrite(save_path, image)