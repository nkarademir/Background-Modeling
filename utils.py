import numpy as np
import cv2


def to_rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


def boxes_intersect(bbox1, bbox2):  # Return if two rect overlap
    return ((np.abs(bbox1[0]-bbox2[0])*2) < (bbox1[2]+bbox2[2])) and \
           ((np.abs(bbox1[1]-bbox2[1])*2) < (bbox1[3]+bbox2[3]))


def rect_similarity2(r1, r2):
    if boxes_intersect(r1, r2):     # Return if r1 and r2 satisfy overlapping criterion
        if similarity_measure_rect(r1, r2) > 0.5:   # return similarity
            return True
        return False
    return False


def similarity_measure_rect(bbox_test, bbox_target):
    # Return similarity measure between two bounding box
    def gen_box(bbox):
        from shapely.geometry import box
        box = box(bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
        return box
    bbtest = gen_box(bbox_test)
    bbtarget = gen_box(bbox_target)
    return bbtarget.intersection(bbtest).area/bbtarget.union(bbtest).area


def norm_correlate(a, v):
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    v = (v - np.mean(v)) / np.std(v)
    return np.correlate(a, v)


def draw_bounding_box(image, bbox):
    # Draw all bounding box inside image as red rectangle
    for s in bbox:
        cv2.rectangle(image, (s[0], s[1]), (s[0]+s[2], s[1]+s[3]), 255, 1)
    return image


def draw_bounding_box2(image, bbox):
    # Draw all bounding box inside image as red rectangle
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), 255, 1)
    return image
