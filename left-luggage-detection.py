import copy
from imutils.video import FPS
import imutils
from static_object import *
from intensity_processing import *


def left_luggage_detection():
    stream = cv2.VideoCapture("/Users/Naciye/Desktop/vtest0.mp4")
    fps = FPS().start()
    first_run = True
    (ret, frame) = stream.read()

    while not ret:
        (ret, frame) = stream.read()
    frame = imutils.resize(frame, width=450)
    (height, width, channel) = frame.shape
    image_shape = (height, width)
    rgb = IntensityProcessing(image_shape)

    bbox_last_frame_proposals = []
    static_objects = []

    while 1:
        (ret, frame) = stream.read()
        if not ret:
            break
        else:
            frame = imutils.resize(frame, width=450)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.dstack([frame, frame, frame])

            rgb.current_frame = frame  # .getNumpy()
            if first_run:
                old_rgb_frame = copy.copy(rgb.current_frame) # old frame is the new frame
                first_run = False

            rgb.compute_foreground_masks(rgb.current_frame)  # compute foreground masks
            rgb.update_detection_aggregator()   # detect if new object proposed

            rgb_proposal_bbox = rgb.extract_proposal_bbox()     # bounding boxes of the areas proposed
            foreground_rgb_proposal = rgb.proposal_foreground   # rgb proposals

            bbox_current_frame_proposals = rgb_proposal_bbox
            final_result_image = rgb.current_frame.copy()

            old_bbox_still_present = check_bbox_not_moved(bbox_last_frame_proposals, bbox_current_frame_proposals,
                                                          old_rgb_frame, rgb.current_frame.copy())

            # add the old bbox still present in the current frame to the bbox detected
            bbox_last_frame_proposals = bbox_current_frame_proposals + old_bbox_still_present
            old_rgb_frame = rgb.current_frame.copy()

            # static object
            # if len(bbox_last_frame_proposals) > 0:  # not on first frame of video
            #     for old in bbox_last_frame_proposals:
            #         old_drawn = False
            #         for curr in static_objects:
            #             if rect_similarity2(curr.bbox_info, old):
            #                 old_drawn = True
            #                 break
            #         if not old_drawn:
            #             owner_frame = rgb.current_frame.copy()
            #             # draw_bounding_box2(owner_frame, old)
            #             static_objects.append(StaticObject(image_shape, old, owner_frame, 0))
            # ##########
            # draw the proposals bbox in the image
            # print(len(bbox_current_frame_proposals), len(static_objects) )

            draw_bounding_box(final_result_image, bbox_current_frame_proposals)
            draw_bounding_box(foreground_rgb_proposal, rgb_proposal_bbox)

            img = rgb.current_frame
            mask_lg = rgb.foreground_mask_long_term
            mask_sh = rgb.foreground_mask_short_term

            long = cv2.bitwise_and(img, rgb.current_frame, mask=mask_lg)
            cv2.imshow("long", long)

            short = cv2.bitwise_and(img, rgb.current_frame, mask=mask_sh)
            cv2.imshow("short", short)

            cv2.imshow('final_result_image', final_result_image)
            cv2.imshow('foreground_rgb_proposal', foreground_rgb_proposal)
            cv2.imshow('frame', frame)
        # k = cv2.waitKey(1)
        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    stream.release()
    cv2.destroyAllWindows()

    for x in static_objects:
        x.print_object()




def check_bbox_not_moved(bbox_last_frame_proposals, bbox_current_frame_proposals, old_frame, current_frame):
    bbox_to_add = []
    if len(bbox_last_frame_proposals) > 0:  # not on first frame of video
        for old in bbox_last_frame_proposals:
            old_drawn = False
            for curr in bbox_current_frame_proposals:
                if rect_similarity2(old, curr):
                    old_drawn = True
                    break
            if not old_drawn:
                # Check if the area defined by the bounding box in the old frame and in the new one is still the same
                old_section = old_frame[old[1]:old[1] + old[3], old[0]:old[0] + old[2]].flatten()
                new_section = current_frame[old[1]:old[1] + old[3], old[0]:old[0] + old[2]].flatten()
                if norm_correlate(old_section, new_section)[0] > 0.9:
                    bbox_to_add.append(old)
    return bbox_to_add


left_luggage_detection()