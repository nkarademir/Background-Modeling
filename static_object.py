from utils import *


class StaticObject:
    def __init__(self, shape_img, bbox, owner_frame, type):
        # shape_img = shape_img + (3, 0)
        self.bbox_info = bbox
        self.owner = owner_frame.copy()
        self.object_type = type

    def print_object(self):
        print(self.bbox_info, self.object_type)
        cv2.imshow("Owner Frame", self.owner)
        cv2.waitKey(1000)
        cv2.destroyWindow("Owner Frame")
