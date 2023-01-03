import cv2
import numpy as np

def matching_bbox(bboxes, x, y):
    for bbox in bboxes:
        if x >= bbox[0] and x <= bbox[2] and y >= bbox[1] and y <= bbox[3]:
            return True, bbox
    return False, None

selected_bbox = None
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global selected_bbox
    if event == cv2.EVENT_LBUTTONDOWN:
        is_match, selected_bbox = matching_bbox(param, x, y)
        if is_match:
            cv2.destroyWindow("detection")
        else:
            print("No Object")

def manual_selection(result_img, bboxes):
    if bboxes == []:
        return None
    cv2.imshow("detection", result_img)
    bboxes = np.array(bboxes)
    cv2.setMouseCallback("detection", on_EVENT_LBUTTONDOWN, param=bboxes)
    cv2.waitKey(1000)
    return selected_bbox