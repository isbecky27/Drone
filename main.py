# Online System
import os, sys, cv2, time, torch
import multiprocessing
import numpy as np
import pandas as pd
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
gfrom utils.torch_utils import time_synchronized
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

## Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seq', type=str, default='boat4')
parser.add_argument('--interval', type=int, default=0) # 0 is all tracking.
parser.add_argument('--threshold', type=float, default=1)
args = parser.parse_args()
seq = args.seq
seq_len = len(os.listdir(f'./data/UAV123/{seq}/'))
interval = args.interval

## Load Ground Truth
with open(f'./data/UAV123/anno/UAV123/{seq}.txt', 'r') as f:
    gt_box = f.read()
gt_box = gt_box.split('\n')

## Create Result Files
result_path = f'./output/UAV123/{seq}'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(f'{result_path}.csv'):
    d = {'seq_name': [], 'seq_len': [], 'interval': [], 'total time': [], 'avg time': []}
    df = pd.DataFrame(data=d)
else:
    df = pd.read_csv(f'{result_path}.csv', index_col=False)

f = open(f'{result_path}_interval{interval}.txt', 'w')
f.write(f'Start Time: {time.time()}\n')

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
f.write(f'Start to Loading Model: {time.time()}\n')
tracker = OSTrack(tracker_cfg, None, threshold=cfg.threshold_conf) # OSTrack
init_model(cfg) # Yolov7
f.write(f'End of Loading Model: {time.time()}\n')

update, total_time = False, 0
for idx, (path, img) in enumerate(dataset, start=1):

    ## Detection
    if (interval != 0 and (idx-1) % interval == 0) or (interval == 0 and idx == 1): 
        time1 = time_synchronized()
        result_img, bboxes = detect_simple(cfg, path, img)
        time2 = time_synchronized()
        total_time += time2 - time1
        update = True
        # print("Detection:", idx)
    if interval == 1:
        continue

    ## Selection Module (use ground truth)
    bboxes = np.array(bboxes)
    bboxes[:, 2] -= bboxes[:, 0]
    bboxes[:, 3] -= bboxes[:, 1]
    box_gt = gt_box[idx-1].split(',')
    if 'NaN' not in box_gt:
        box_gt = [int(float(i)) for i in box_gt]
        target = selection_bbox(bboxes, box_gt, cfg.threshold_iou)
        if target is None:
            target = box_gt
        bbox = {'init_bbox': target}

    ## Tracker
    if update and (target is not None):
        update = False
        time1 = time_synchronized()
        tracker.initialize(img, bbox)
        time2 = time_synchronized()
        total_time += time2 - time1
        # print("Initialize template:", idx)
        continue
    time1 = time_synchronized()
    output = tracker.track(img)
    time2 = time_synchronized()
    total_time += time2 - time1
    # print("Tracking:", idx)
    update = output['update_template']

    ## Show Results
    # box = output['target_bbox']
    # box = [int(float(i)) for i in box]
    # color = (255, 0, 0)
    # image = cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, thickness = 2)
    # p = Path(path)  # to Path
    # save_path = str(save_dir / p.name)
    # cv2.imwrite(save_path, image)

f.write(f'End Time: {time.time()}\n')
f.close()
df.loc[len(df.index)] = [seq, seq_len, interval, total_time, total_time / seq_len]
df.to_csv(f'{result_path}.csv', index=False)