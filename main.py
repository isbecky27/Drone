# Online System
import os, sys, cv2, torch
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
from detect import detect
from yolov7.utils.torch_utils import time_synchronized

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
parser.add_argument('--interval', type=int, default=0) # 0 is all tracking.
parser.add_argument('--threshold', type=float, default=1)
args = parser.parse_args()

## Get Parameters
cfg = get_params('./config.yaml')
tracker_cfg = get_parameters(cfg.tracker, cfg.tracker_params)

## Initialize
tracker = OSTrack(tracker_cfg, None, threshold=1.0)

seq = 'boat4'
seq_len = len(os.listdir(f'./data/UAV123/{seq}/'))
interval = args.interval
update, total_time = False, 0

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

for idx in range(1, seq_len + 1):

    ## Detection
    if (interval != 0 and (idx-1) % interval == 0) or (interval == 0 and idx == 1): 
        cfg.source = f'./data/UAV123/{seq}/{str(idx).zfill(6)}.jpg'
        result_img, bboxes, inference_time = detect(cfg)
        total_time += inference_time
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
        target = selection_bbox(bboxes, box_gt)
        bbox = {'init_bbox': target}

    ## Tracker
    img = cv2.imread(f'./data/{seq}/{str(idx).zfill(6)}.jpg')
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
    # cv2.imwrite(f'{result_path}/{str(idx).zfill(6)}.jpg', image)

df.loc[len(df.index)] = [seq, seq_len, interval, total_time, total_time / seq_len]
df.to_csv(f'{result_path}.csv', index=False)