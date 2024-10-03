# -*- coding: utf-8 -*-

# from collections import deque
from typing import List

import argparse
import configparser
import numpy as np
import cv2
import os
import math
import subprocess
from pathlib import Path
import isSameIMAGE


# -------------------Global Parameters Start-----------------------#
# -----------------Parameters that affect the result.---------------#
THICKNESS_TEXT = 1
COLOR_TEXT = (255, 255, 255)  # white
textOutputPosRow1 = (1, 30)
textOutputPosRow2 = (1, 50)
textOutputPosRow3 = (1, 70)

rRate = 0.05  # Not used, spare
sRate = 0.1  # If the change in area is within ±?%, it is recognized as the same object.

# If the color range is within the following range, it will be considered as a substance.
# Lower = np.array([26, 43, 46])
Lower = np.array([10, 10, 10])
Upper = np.array([99, 255, 255])
THRESHOLD1 = 10  # Canny's THRESHOLD1
THRESHOLD2 = 127  # Canny's THRESHOLD2

SAME_STATIC_FRAME_MIN_SPACING = 10

blurKernel = (5, 5)
sigmaImg = 5

# If the deviation range of left, right, up and down is within the range, it is judged that there is no movement.
xPixelDeviation = 2.0
yPixelDeviation = 2.0

"""
windowWidth = 640
xPixelShift = 1920 - windowWidth - 440  xPixelShift = 1920 - windowWidth - 440  # X-axis coordinate of the start of the extraction window
putTextFontSize = 2.0
"""

# binning's video
windowWidth = 320
xPixelShift = (
    960 - windowWidth - 220
)  # X-axis coordinate of the start of the extraction window
putTextFontSize = 0.8

extendPixel = 200  # When performing secondary verification, pay attention to whether the pixels expanded to the left and right of the extraction window exceed the range of the image!!!!
distance_max = 100  # The maximum distance between the center of the partial particles entering the extraction window and the particles entering the extended extraction window.

videoSpeed = 1  # How many times

# Valid radius range
minRadius = 2
maxRadius = 250
# S size
smallSize = 25
smallSizeColor = (0, 255, 255)  # yellow
# M size
middleSize = 50
middleSizeColor = (0, 0, 255)  # red
# L size
largeSize = 75
largeSizeColor = (0, 255, 0)  # green
# XL size
# larger than X size
xLargeSizeColor = (255, 0, 0)  # blue
sizeFrameColor = (0, 200, 200)

# Distinction condition between fiber type and grain type, ?% of the area of the circumscribed circle
areaRatio = 0.3

prefixFilename = "pic"
extName = ".jpg"
tmpPic = ".\\data\\pic\\"  # Intermediate folder for storing input video as images
tmpCropPic = ".\\data\\picCrop\\"
tmpCropExtPic = ".\\data\\picCropExt\\"
tmpRapidPic = ".\\data\\rapid\\"

flagTmpPic = (
    False  # If True: Convert input video to images and place in intermediate folder
)
tmpPic2 = ".\\data\\pic2\\"  # Intermediate folder for storing output video as images
flagTmpPic2 = (
    False  # If True: Convert output video to images and place in intermediate folder
)
flg = True  # If True: Output frames even if there are no objects.
flagTmpPicRapid = False
flagIsOutputVideo = False

staticFrameJudgeMethodFlag = False  # True:PHASH()  False:absdiff()

flagDoRapid = True

outputPathAndFilePrefix = ".\\output\\out_"
# -------------------Global Parameters End-----------------------#


class MicroPlasticObj:
    def __init__(self, x, y, r, s, contour):
        self.x = x
        self.y = y
        self.r = r  # R of the circumscribed circle
        self.s = s  # Area of the polygon
        self.contour = contour

    def is_same_obj(self, new_obj):
        if self.x - xPixelDeviation > new_obj.x:
            return False
        if new_obj.y - yPixelDeviation < self.y < new_obj.y + yPixelDeviation:
            return True
        else:
            return False
        return True


def do_proc(cmd):
    completedProcess = subprocess.run(
        cmd,
        check=True,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )


def do_rapid(targetPath, db_folder):
    # targetPath: Full path of the directory containing the images

    p = Path(targetPath)
    dbpath = Path(db_folder)
    # Extract only jpg files
    image_path_list = [x for x in p.glob("*.jpg") if x.is_file()]
    labelname = "label1"
    groupname = "MicroPlastic"
    modelname = "MicroPlastic"
    labelfilename = p.resolve().as_posix() + "/label.csv"
    with open(labelfilename, "wt") as label_file:
        # Header
        # File path where the images are located=file_path
        label_file.write(f'[DataPath]"{p.resolve().as_posix()}\\"\n')
        # image_path_list: List of files taken from an arbitrary folder
        # If an error occurs, place an appropriate label after the last comma or remove the comma.
        for image_path in image_path_list:
            label_file.write(f'2,"{image_path.name}"\n')

    # ラベル登録
    cmd = [
        "rapid",
        "/S",
        "IMA/manage_data.rpd",
        "label-import",
        "-gl",
        f"{{{groupname},{labelname},{labelfilename},}}",
        "-ltype",
        "image",
        "-path",
        str(dbpath.resolve()),
    ]
    do_proc(cmd)

    # Execute classification
    cmd = [
        "rapid",
        "/S",
        "IMA/exec_classify.rpd",
        "-p",
        "1",
        "-t",
        "CNN",
        "-m",
        modelname,
        "-l",
        labelname,  #'-param', 'Full path of the analysis config file',
        "-nodb",
        "-o",
        p.resolve().as_posix() + "/results.csv",
        "-path",
        str(dbpath.resolve()),
    ]
    do_proc(cmd)

    # Delete label
    cmd = [
        "rapid",
        "/S",
        "IMA/manage_data.rpd",
        "label-delete",
        "-l",
        labelname,
        "-path",
        str(dbpath.resolve()),
    ]
    do_proc(cmd)

    return


def update_obj_list(
    old_list: List[MicroPlasticObj],
    new_list: List[MicroPlasticObj],
    move_out_total_list: List[MicroPlasticObj],
):
    move_in_list: List[MicroPlasticObj] = []

    if len(new_list) == 0 and len(old_list) != 0:
        # Currently, this branch is not executed
        # Simple case where all objects have moved out
        move_out_total_list.extend(
            old_list
        )  # All particles that moved out of the screen
        old_list.clear()
    else:
        for newObj in new_list:
            temp_flg = False
            for oldObj in old_list:
                if oldObj.is_same_obj(newObj):
                    old_list.remove(oldObj)
                    temp_flg = True
                    break
                # Do nothing in the else case
            if not temp_flg:  # The same particle is not found
                move_in_list.append(
                    newObj
                )  # Particles that move into the screen this time
        move_out_total_list.extend(
            old_list
        )  # All particles that moved out of the screen
    return move_in_list


# Use the contour as a mask to cut out only one microplastic.
def contours_cutout(
    obj_center,
    frame,
    cnt_frame,
    index,
    center,
    radius,
    is_fiber_flg,
    micro_plastic_size,
):

    radius_str_r = "r" + str(round(radius, 1))
    radius_str = str(round(radius, 1))

    if is_fiber_flg:
        str_kind = "fiber"
        filename_tmp = (
            tmpRapidPicFiber
            + "\\"
            + prefixFilename
            + str(cnt_frame).zfill(8)
            + "_"
            + str(index).zfill(3)
            + "_"
            + str_kind
            + "_size_"
            + micro_plastic_size
            + "_"
            + radius_str_r
            + ".jpg"
        )
    else:
        str_kind = "grain"
        filename_tmp = (
            tmpRapidPicGrain
            + "\\"
            + prefixFilename
            + str(cnt_frame).zfill(8)
            + "_"
            + str(index).zfill(3)
            + "_"
            + str_kind
            + "_size_"
            + micro_plastic_size
            + "_"
            + radius_str_r
            + ".jpg"
        )

    just_filename = (
        prefixFilename
        + str(cnt_frame).zfill(8)
        + "_"
        + str(index).zfill(3)
        + "_"
        + str_kind
        + "_size_"
        + micro_plastic_size
        + "_"
        + radius_str_r
        + ".jpg"
    )

    frame_tmp = frame.copy()
    # Create a blank canvas
    mask = np.tile(np.uint8(0), (frame_tmp.shape[0], frame_tmp.shape[1], 1))
    # Get the contours with the largest area
    mask = cv2.drawContours(mask, [obj_center], 0, 255, -1)

    # Use the mask to extract the original image
    sub_img = cv2.bitwise_or(frame_tmp, frame_tmp, mask=mask)
    if radius < 50:
        radius = 50
    else:
        radius = int(radius) + 1

    # Extract a square from the center to the radius, and adjust if it extends outside the entire image.
    # x0, x1, y0, y1
    if (center[0] - radius) < 0:
        x0 = 0
        x1 = 2 * radius
    elif center[0] + radius > frame_tmp.shape[1]:
        x0 = frame_tmp.shape[1] - 2 * radius
        x1 = frame_tmp.shape[1]
    else:
        x0 = center[0] - radius
        x1 = center[0] + radius

    if (center[1] - radius) < 0:
        y0 = 0
        y1 = 2 * radius
    elif center[1] + radius > frame_tmp.shape[0]:
        y0 = frame_tmp.shape[0] - 2 * radius
        y1 = frame_tmp.shape[0]
    else:
        y0 = center[1] - radius
        y1 = center[1] + radius

    # print(y0, y1, x0, x1, center, radius)
    sub_img = sub_img[y0:y1, x0:x1]
    cv2.imwrite(filename_tmp, sub_img)  # Save as image
    csvList.write(
        just_filename
        + ","
        + radius_str
        + ","
        + micro_plastic_size
        + ","
        + str_kind
        + "\n"
    )
    return


def video_marker(mp_filename: str):

    video_file = os.path.join(
        inputPath, mp_filename
    )  # Input video storage location and file name
    output_file = os.path.join(
        output_folder, "out_" + mp_filename
    )  # Output video storage location and file name

    # Create an intermediate file storage location if it does not exist. START
    if flagTmpPic:
        if not os.path.exists(tmpPic):
            os.makedirs(tmpPic)
        if not os.path.exists(tmpCropPic):
            os.makedirs(tmpCropPic)
        if not os.path.exists(tmpCropExtPic):
            os.makedirs(tmpCropExtPic)

    if flagTmpPic2:
        if not os.path.exists(tmpPic2):
            os.makedirs(tmpPic2)

    if flagTmpPicRapid:
        if not os.path.exists(tmpRapidPic):
            os.makedirs(tmpRapidPic)

    # Create an intermediate file storage location if it does not exist. END

    vc = cv2.VideoCapture(video_file)
    output_fps = (
        vc.get(cv2.CAP_PROP_FPS) * videoSpeed
    )  # If increased, the video will be fast-forwarded at videoSpeed times the normal speed
    if flagIsOutputVideo:
        video_writer = cv2.VideoWriter(
            output_file,
            cv2.VideoWriter_fourcc("X", "V", "I", "D"),
            output_fps,
            (
                int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )
    cnt_frame = 0
    sum_objects = 0
    is_first_img = True  # flag: is the first image by read in video
    p_score_old = 0.0  # Initialize the image matching score calculated by p hash
    fiber_small_size_cnt = 0
    grain_small_size_cnt = 0
    fiber_middle_size_cnt = 0
    grain_middle_size_cnt = 0
    fiber_large_size_cnt = 0
    grain_large_size_cnt = 0
    fiber_x_large_size_cnt = 0
    grain_x_large_size_cnt = 0
    static_frame_old = -10

    while True:  # Read frames in a loop
        cnt_frame = cnt_frame + 1
        tmp_pic_file_name = tmpPic + prefixFilename + str(cnt_frame).zfill(8) + ".jpg"
        tmp_crop_pic_file_name = (
            tmpCropPic + prefixFilename + str(cnt_frame).zfill(8) + ".jpg"
        )
        # tmp_crop_ext_pic_file_name = tmpCropExtPic + prefixFilename + str(cnt_frame).zfill(8) + '.jpg'
        filename = prefixFilename + str(cnt_frame).zfill(8)

        ret_val, frame = vc.read()
        if ret_val:
            if is_first_img:
                is_first_img = False
                frame_old = frame
                is_the_same_img = False
                flg_tmp_old = False
                cnt_same = 0
                # output_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
                output_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                if staticFrameJudgeMethodFlag:
                    p_hash_str1 = isSameIMAGE.pHash(frame_old)
                    p_hash_str2 = isSameIMAGE.pHash(frame)
                    p_score = 1 - isSameIMAGE.hammingDist(
                        p_hash_str1, p_hash_str2
                    ) * 1.0 / (32 * 32 / 4)
                    if p_score == 1.0:
                        if p_score_old == 1.0:  # or p_score_old == 0.0:
                            cnt_same = cnt_same + 1
                            is_the_same_img = True
                        else:
                            cnt_same = 0
                            is_the_same_img = False
                    else:
                        cnt_same = 0
                        is_the_same_img = False
                    p_score_old = p_score
                    frame_old = frame
                else:
                    flg_tmp = isSameIMAGE.is_same_frame_by_diff(frame_old, frame)
                    cnt_same = 2  # cnt_same has no meaning in this branch
                    if flg_tmp and (not flg_tmp_old):
                        is_the_same_img = True
                    else:
                        is_the_same_img = False
                    flg_tmp_old = flg_tmp
                    frame_old = frame

            # [y:y1, x:x1], Window range
            # Use when debugging ↓
            # cv2.rectangle(frame, (xPixelShift, 0), ((xPixelShift + windowWidth), output_height), (0, 0, 255), 5)
            if flagTmpPic:
                cv2.imwrite(tmp_pic_file_name, frame)  # Save as image
                print(tmp_pic_file_name)
                frame_crop = frame[
                    0:output_height, xPixelShift : (xPixelShift + windowWidth)
                ]
                cv2.imwrite(tmp_crop_pic_file_name, frame_crop)  # Save as image
        else:
            break

        if (
            is_the_same_img
            and cnt_same >= 2
            and (cnt_frame - static_frame_old) >= SAME_STATIC_FRAME_MIN_SPACING
        ):
            # Store the frame number of the previous static image
            static_frame_old = cnt_frame
            # Get contours
            contours_objects = get_contours_from_frame(frame, 2)
            # Check if contours exist
            if len(contours_objects) > 0:

                frame1 = frame.copy()
                frame_flash = frame.copy()
                # Draw a red frame for the ROI area
                cv2.rectangle(
                    frame1,
                    (xPixelShift, 0),
                    ((xPixelShift + windowWidth), output_height),
                    (0, 0, 255),
                    1,
                )
                cv2.rectangle(
                    frame_flash,
                    (xPixelShift, 0),
                    ((xPixelShift + windowWidth), output_height),
                    (0, 0, 255),
                    1,
                )

                new_obj_list = []
                for index, objCenter in enumerate(contours_objects):
                    # Determine the circumscribed circle of the contour with the largest area
                    ((x, y), radius) = cv2.minEnclosingCircle(objCenter)
                    # Calculate the moments of the contour
                    M = cv2.moments(objCenter)
                    area = cv2.contourArea(objCenter)
                    # Calculate the centroid
                    if (M["m00"] == 0.0) or (M["m00"] == 0.0):
                        center = (int(x), int(y))
                        # print(center, "11111111111111111")
                    else:
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        # print(center, "22222222222222222")

                    # print("----"+ center + "----" + file )
                    # print(center, area, radius)
                    # print(tmp_pic_file_name)

                    if maxRadius > radius > minRadius and xPixelShift < center[0] < (
                        xPixelShift + windowWidth
                    ):

                        if area / (math.pi * radius * radius) < areaRatio:
                            # fiber
                            is_fiber_flg = True
                        else:
                            # grain
                            is_fiber_flg = False

                        # Determine the size based on the radius:
                        if radius <= smallSize:
                            micro_plastic_size = "S"
                            tmp_color = smallSizeColor
                            if is_fiber_flg:
                                fiber_small_size_cnt = fiber_small_size_cnt + 1
                            else:
                                grain_small_size_cnt = grain_small_size_cnt + 1
                        elif radius <= middleSize:
                            micro_plastic_size = "M"
                            tmp_color = middleSizeColor
                            if is_fiber_flg:
                                fiber_middle_size_cnt = fiber_middle_size_cnt + 1
                            else:
                                grain_middle_size_cnt = grain_middle_size_cnt + 1
                        elif radius <= largeSize:
                            micro_plastic_size = "L"
                            tmp_color = largeSizeColor
                            if is_fiber_flg:
                                fiber_large_size_cnt = fiber_large_size_cnt + 1
                            else:
                                grain_large_size_cnt = grain_large_size_cnt + 1
                        else:
                            micro_plastic_size = "XL"
                            tmp_color = xLargeSizeColor
                            if is_fiber_flg:
                                fiber_x_large_size_cnt = fiber_x_large_size_cnt + 1
                            else:
                                grain_x_large_size_cnt = grain_x_large_size_cnt + 1

                        # If the radius is within a certain range, it is a counting target
                        #
                        if is_fiber_flg:
                            cv2.rectangle(
                                frame1,
                                (int(x) - int(radius), int(y) - int(radius)),
                                (int(x) + int(radius), int(y) + int(radius)),
                                tmp_color,
                                1,
                            )
                        else:
                            cv2.circle(
                                frame1, (int(x), int(y)), int(radius), tmp_color, 1
                            )  # Draw the circumscribed circle
                        # cv2.drawContours(frame1, contours_objects, index, (0, 200, 200), 1)  # Draw the contour
                        # cv2.circle(frame1, center, 5, (0, 0, 255), -1)  # Center point
                        contours_cutout(
                            objCenter,
                            frame,
                            cnt_frame,
                            index,
                            center,
                            radius,
                            is_fiber_flg,
                            micro_plastic_size,
                        )

                        # To create a flash effect, change the color, draw a circumscribed circle for the detected object, and draw the center point
                        if is_fiber_flg:
                            cv2.rectangle(
                                frame1,
                                (int(x) - int(radius), int(y) - int(radius)),
                                (int(x) + int(radius), int(y) + int(radius)),
                                tmp_color,
                                1,
                            )
                        else:
                            cv2.circle(
                                frame_flash, (int(x), int(y)), int(radius), tmp_color, 1
                            )  # Draw the circumscribed circle
                        # cv2.drawContours(frame_flash, contours_objects, index, (0, 255, 255), 1)  # Draw the contour
                        # cv2.circle(frame_flash, center, 5, (0, 0, 255), -1)  # Center point

                        # Add detected objects to the list
                        new_obj_list.append(
                            MicroPlasticObj(
                                center[0], center[1], radius, area, objCenter
                            )
                        )

                sum_objects = sum_objects + len(new_obj_list)
                str_msg_row_1 = "The %06d frames. count in ROI:%03d. sum:%06d" % (
                    cnt_frame,
                    len(new_obj_list),
                    sum_objects,
                )
                sum_fiber = (
                    fiber_small_size_cnt
                    + fiber_middle_size_cnt
                    + fiber_large_size_cnt
                    + fiber_x_large_size_cnt
                )
                str_msg_row_2 = "Fiber. SUM:%06d. S:%06d. M:%06d. L:%06d. XL:%06d. " % (
                    sum_fiber,
                    fiber_small_size_cnt,
                    fiber_middle_size_cnt,
                    fiber_large_size_cnt,
                    fiber_x_large_size_cnt,
                )
                sum_grain = (
                    grain_small_size_cnt
                    + grain_middle_size_cnt
                    + grain_large_size_cnt
                    + grain_x_large_size_cnt
                )
                str_msg_row_3 = "Grain. SUM:%06d. S:%06d. M:%06d. L:%06d. XL:%06d. " % (
                    sum_grain,
                    grain_small_size_cnt,
                    grain_middle_size_cnt,
                    grain_large_size_cnt,
                    grain_x_large_size_cnt,
                )

                print(str_msg_row_1, str_msg_row_2, str_msg_row_3)
                frame2 = frame1.copy()
                # Write text on the image
                cv2.putText(
                    frame2,
                    str_msg_row_1,
                    textOutputPosRow1,
                    cv2.FONT_HERSHEY_PLAIN,
                    putTextFontSize,
                    COLOR_TEXT,
                    THICKNESS_TEXT,
                )
                cv2.putText(
                    frame2,
                    str_msg_row_2,
                    textOutputPosRow2,
                    cv2.FONT_HERSHEY_PLAIN,
                    putTextFontSize,
                    COLOR_TEXT,
                    THICKNESS_TEXT,
                )
                cv2.putText(
                    frame2,
                    str_msg_row_3,
                    textOutputPosRow3,
                    cv2.FONT_HERSHEY_PLAIN,
                    putTextFontSize,
                    COLOR_TEXT,
                    THICKNESS_TEXT,
                )

                cv2.putText(
                    frame_flash,
                    str_msg_row_1,
                    textOutputPosRow1,
                    cv2.FONT_HERSHEY_PLAIN,
                    putTextFontSize,
                    COLOR_TEXT,
                    THICKNESS_TEXT,
                )
                cv2.putText(
                    frame_flash,
                    str_msg_row_2,
                    textOutputPosRow2,
                    cv2.FONT_HERSHEY_PLAIN,
                    putTextFontSize,
                    COLOR_TEXT,
                    THICKNESS_TEXT,
                )
                cv2.putText(
                    frame_flash,
                    str_msg_row_3,
                    textOutputPosRow3,
                    cv2.FONT_HERSHEY_PLAIN,
                    putTextFontSize,
                    COLOR_TEXT,
                    THICKNESS_TEXT,
                )

                if flagTmpPic2:
                    cv2.imwrite(tmpPic2 + filename + "1" + "." + extName, frame2)
                if flagIsOutputVideo:
                    video_writer.write(frame2)

                if flagTmpPic2:
                    cv2.imwrite(tmpPic2 + filename + "2" + "." + extName, frame_flash)
                    #
                    cv2.imwrite(tmpPic2 + filename + "3" + "." + extName, frame2)
                    cv2.imwrite(tmpPic2 + filename + "4" + "." + extName, frame_flash)
                    cv2.imwrite(tmpPic2 + filename + "5" + "." + extName, frame2)
                if flagIsOutputVideo:
                    video_writer.write(frame_flash)
                    video_writer.write(frame2)
                    video_writer.write(frame_flash)
                    video_writer.write(frame2)

            elif flg:
                str_msg_row_1 = "The %06d frames. count in ROI:%03d. sum:%06d" % (
                    cnt_frame,
                    0,
                    sum_objects,
                )
                sum_fiber = (
                    fiber_small_size_cnt
                    + fiber_middle_size_cnt
                    + fiber_large_size_cnt
                    + fiber_x_large_size_cnt
                )
                str_msg_row_2 = "Fiber. SUM:%06d. S:%06d. M:%06d. L:%06d. XL:%06d. " % (
                    sum_fiber,
                    fiber_small_size_cnt,
                    fiber_middle_size_cnt,
                    fiber_large_size_cnt,
                    fiber_x_large_size_cnt,
                )
                sum_grain = (
                    grain_small_size_cnt
                    + grain_middle_size_cnt
                    + grain_large_size_cnt
                    + grain_x_large_size_cnt
                )
                str_msg_row_3 = "Grain. SUM:%06d. S:%06d. M:%06d. L:%06d. XL:%06d. " % (
                    sum_grain,
                    grain_small_size_cnt,
                    grain_middle_size_cnt,
                    grain_large_size_cnt,
                    grain_x_large_size_cnt,
                )

                print(str_msg_row_1, str_msg_row_2, str_msg_row_3)
                frame2 = frame.copy()
                # Write text on the image
                cv2.putText(
                    frame2,
                    str_msg_row_1,
                    textOutputPosRow1,
                    cv2.FONT_HERSHEY_PLAIN,
                    putTextFontSize,
                    COLOR_TEXT,
                    THICKNESS_TEXT,
                )
                cv2.putText(
                    frame2,
                    str_msg_row_2,
                    textOutputPosRow2,
                    cv2.FONT_HERSHEY_PLAIN,
                    putTextFontSize,
                    COLOR_TEXT,
                    THICKNESS_TEXT,
                )
                cv2.putText(
                    frame2,
                    str_msg_row_3,
                    textOutputPosRow3,
                    cv2.FONT_HERSHEY_PLAIN,
                    putTextFontSize,
                    COLOR_TEXT,
                    THICKNESS_TEXT,
                )

                # Draw a red frame for the ROI area
                cv2.rectangle(
                    frame2,
                    (xPixelShift, 0),
                    ((xPixelShift + windowWidth), output_height),
                    (0, 0, 255),
                    1,
                )

                if flagTmpPic2:
                    cv2.imwrite(tmpPic2 + filename + "0" + "." + extName, frame2)
                if flagIsOutputVideo:
                    video_writer.write(frame2)

        elif flg:
            str_msg_row_1 = "The %06d frames. count in ROI:%03d. sum:%06d" % (
                cnt_frame,
                0,
                sum_objects,
            )
            sum_fiber = (
                fiber_small_size_cnt
                + fiber_middle_size_cnt
                + fiber_large_size_cnt
                + fiber_x_large_size_cnt
            )
            str_msg_row_2 = "Fiber. SUM:%06d. S:%06d. M:%06d. L:%06d. XL:%06d. " % (
                sum_fiber,
                fiber_small_size_cnt,
                fiber_middle_size_cnt,
                fiber_large_size_cnt,
                fiber_x_large_size_cnt,
            )
            sum_grain = (
                grain_small_size_cnt
                + grain_middle_size_cnt
                + grain_large_size_cnt
                + grain_x_large_size_cnt
            )
            str_msg_row_3 = "Grain. SUM:%06d. S:%06d. M:%06d. L:%06d. XL:%06d. " % (
                sum_grain,
                grain_small_size_cnt,
                grain_middle_size_cnt,
                grain_large_size_cnt,
                grain_x_large_size_cnt,
            )

            print(str_msg_row_1, str_msg_row_2, str_msg_row_3)
            frame2 = frame.copy()
            # Write text on the image
            cv2.putText(
                frame2,
                str_msg_row_1,
                textOutputPosRow1,
                cv2.FONT_HERSHEY_PLAIN,
                putTextFontSize,
                COLOR_TEXT,
                THICKNESS_TEXT,
            )
            cv2.putText(
                frame2,
                str_msg_row_2,
                textOutputPosRow2,
                cv2.FONT_HERSHEY_PLAIN,
                putTextFontSize,
                COLOR_TEXT,
                THICKNESS_TEXT,
            )
            cv2.putText(
                frame2,
                str_msg_row_3,
                textOutputPosRow3,
                cv2.FONT_HERSHEY_PLAIN,
                putTextFontSize,
                COLOR_TEXT,
                THICKNESS_TEXT,
            )
            # ROI区の赤枠を描く
            cv2.rectangle(
                frame2,
                (xPixelShift, 0),
                ((xPixelShift + windowWidth), output_height),
                (0, 0, 255),
                1,
            )

            if flagTmpPic2:
                cv2.imwrite(tmpPic2 + filename + "0" + "." + extName, frame2)
            if flagIsOutputVideo:
                video_writer.write(frame2)

    # Output CSV summary
    str_msg_row_1 = "%06d frames,SUM,S,M,L,XL" % (cnt_frame,)
    sum_fiber = (
        fiber_small_size_cnt
        + fiber_middle_size_cnt
        + fiber_large_size_cnt
        + fiber_x_large_size_cnt
    )
    str_msg_row_2 = "Fiber,%06d,%06d,%06d,%06d,%06d" % (
        sum_fiber,
        fiber_small_size_cnt,
        fiber_middle_size_cnt,
        fiber_large_size_cnt,
        fiber_x_large_size_cnt,
    )
    sum_grain = (
        grain_small_size_cnt
        + grain_middle_size_cnt
        + grain_large_size_cnt
        + grain_x_large_size_cnt
    )
    str_msg_row_3 = "Grain,%06d,%06d,%06d,%06d,%06d" % (
        sum_grain,
        grain_small_size_cnt,
        grain_middle_size_cnt,
        grain_large_size_cnt,
        grain_x_large_size_cnt,
    )
    csvSum.write(str_msg_row_1 + "\n" + str_msg_row_2 + "\n" + str_msg_row_3 + "\n")

    vc.release()
    if flagIsOutputVideo:
        video_writer.release()


def is_object_stop(new_list: List[MicroPlasticObj], old_list: List[MicroPlasticObj]):
    if (new_list is None) or (old_list is None):
        return False
    if len(new_list) >= 3 and len(old_list) >= 3:
        if (
            new_list[0].is_same_obj(old_list[0])
            and new_list[1].is_same_obj(old_list[1])
            and new_list[2].is_same_obj(old_list[2])
        ):
            return True
        else:
            return False
    if len(new_list) >= 2 and len(old_list) >= 2:
        if new_list[0].is_same_obj(old_list[0]) and new_list[1].is_same_obj(
            old_list[1]
        ):
            return True
        else:
            return False
    if len(new_list) >= 1 and len(old_list) >= 1:
        if new_list[0].is_same_obj(old_list[0]):
            return True
        else:
            return False


def get_micro_plastic_obj_list(contours_objects_in_crop, pixel_shift):
    # The coordinates of the center points of each Obj in the returned List have been calibrated.
    new_obj_list = []
    for objCenter in contours_objects_in_crop:
        # Determine the circumscribed circle of the contour with the largest area
        ((x, y), radius) = cv2.minEnclosingCircle(objCenter)
        # Calculate the moments of the contour
        M = cv2.moments(objCenter)
        area = cv2.contourArea(objCenter)
        # Calculate the centroid
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # Convert from centroid of crop to entire image: x + offset
        center = (center[0] + pixel_shift, center[1])

        if maxRadius > radius > minRadius:
            # Add detected objects to the list
            new_obj_list.append(
                MicroPlasticObj(center[0], center[1], radius, area, objCenter)
            )
    return new_obj_list


def get_contours_from_frame(frame_crop, flag: int = 1):
    # flag =1 Detection using HSV ⇒ GaussianBlur ⇒ Binarization ⇒ Erosion ⇒ Dilation
    # flag =2 Detection using Gray ⇒ Canny ⇒ Dilation ⇒ Erosion
    if flag == 1:
        # Focus on particles that entered the ROI area
        hsv = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, blurKernel, sigmaX=sigmaImg)
        # Create a mask based on the threshold
        mask = cv2.inRange(hsv, Lower, Upper)
        # Erosion operation
        mask = cv2.erode(mask, None, iterations=2)
        # Dilation operation, actually eroding first and then dilating has the effect of opening operation, removing noise
        mask = cv2.dilate(mask, None, iterations=2)
        # Contour detection
        contours_objects = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[-2]
    else:
        # Focus on particles that entered the ROI area
        image = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
        image = cv2.Canny(image, THRESHOLD1, THRESHOLD2)
        # Dilation operation, actually eroding first and then dilating has the effect of opening operation, removing noise
        image = cv2.dilate(image, None, iterations=2)
        # Erosion operation
        image = cv2.erode(image, None, iterations=2)
        # Contour detection
        contours_objects = cv2.findContours(
            image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[-2]
    return contours_objects


def duplicate_del(input_path, output_path):

    i = 0
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    list_img = []
    for root, dirs, files in os.walk(input_path, topdown=False):
        pass
    for file in files:
        raw_img = os.path.join(root, file)
        img1 = cv2.imread(raw_img)
        list_img.append(img1)
    for i in range(len(list_img)):
        p_hash_str1 = isSameIMAGE.pHash(list_img[i])

        if i + 1 >= len(list_img):
            break
        else:
            z = i + 1
        while True:
            p_hash_str2 = isSameIMAGE.pHash(list_img[z])
            p_score = 1 - isSameIMAGE.hammingDist(p_hash_str1, p_hash_str2) * 1.0 / (
                32 * 32 / 4
            )
            if p_score >= 0.98:
                flag = True
                break
            else:
                flag = False
                if z + 1 >= len(list_img):
                    break
                else:
                    z = z + 1
        if not flag:
            filename_tmp = output_path + str(i).zfill(8) + ".jpg"
            cv2.imwrite(filename_tmp, list_img[i])  # Save as image

    filename_tmp = output_path + str(i).zfill(8) + ".jpg"
    cv2.imwrite(filename_tmp, list_img[i])  # Save as image
    return


if "__main__" == __name__:

    # Parameter parsing process
    parser = argparse.ArgumentParser(
        description="this tool is a detector for the micro plastic."
        "It's can classify and count."
        ".\\microPlasticConfig.ini is must be placed."
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="a string for input video path and filename.",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        required=False,
        help="a string for output path. default is ./output. "
        "output csv, video file is placed under it."
        "csv default name is image_info_list.csv and summary_info.csv."
        "video file default name is out_[input filename]_[timestamp",
    )
    parser.add_argument(
        "-m",
        "--image_folder",
        type=str,
        required=False,
        help="a string for output micro plastic cropped images path. "
        "default is ./output/image.",
    )
    parser.add_argument(
        "-d",
        "--db_folder",
        type=str,
        required=False,
        help="a string for rapid meta database path. " "default is ./DB",
    )

    args = parser.parse_args()

    # Check if the input file exists
    if args.input:
        assert os.path.exists(args.input), "input file {} is not exist".format(
            args.input
        )

    if args.output_folder:
        output_folder = args.output_folder
    else:
        output_folder = ".\\output"

    if args.image_folder:
        image_folder = args.image_folder
    else:
        image_folder = os.path.join(output_folder, "image")

    if args.db_folder:
        db_folder = args.db_folder
    else:
        db_folder = ".\\DB"

    # Check if the output folder exists
    if os.path.exists(output_folder):
        pass
    else:
        os.makedirs(output_folder, exist_ok=True)

    config = configparser.ConfigParser()

    print("--- Load config file ---")
    config.read("./microPlasticConfig.ini")

    # returned sections list is not have [default]
    print("> config sections : %s" % config.sections())

    # is have sections in config file
    print("> Load config file is :")

    for section in config.keys():
        if section == "DEFAULT":
            continue
        print("[{s}]".format(s=section))
        for key in config[section]:
            print("{k} = {v}".format(k=key, v=config[section][key]))

    # Set configuration information from config file
    sRate = float(config["param"]["sRate"])
    videoWindowWidth = int(config["param"]["videoWindowWidth"])
    cropWindowWidth = int(config["param"]["cropWindowWidth"])
    cropWindowRightSideDistance = int(config["param"]["cropWindowRightSideDistance"])
    THRESHOLD1 = int(config["param"]["THRESHOLD1"])
    THRESHOLD2 = int(config["param"]["THRESHOLD2"])
    SAME_STATIC_FRAME_MIN_SPACING = int(
        config["param"]["same_static_frame_min_spacing"]
    )
    xPixelDeviation = float(config["param"]["xPixelDeviation"])
    yPixelDeviation = float(config["param"]["yPixelDeviation"])
    videoSpeed = float(config["param"]["videoSpeed"])
    flagIsOutputVideo = config["param"]["outputVideoFlag"]
    if flagIsOutputVideo.lower() in ("yes", "true", "t", "1"):
        flagIsOutputVideo = True
    else:
        flagIsOutputVideo = False
    minRadius = int(config["param"]["minRadius"])
    maxRadius = int(config["param"]["maxRadius"])
    smallSize = int(config["param"]["smallSize"])
    middleSize = int(config["param"]["middleSize"])
    largeSize = int(config["param"]["largeSize"])

    windowWidth = cropWindowWidth
    xPixelShift = (
        videoWindowWidth - cropWindowWidth - cropWindowRightSideDistance
    )  # Starting X-axis coordinate of the extraction window

    inputPath = os.path.dirname(args.input)
    inputFile = os.path.basename(args.input)
    tmpRapidPicFiber = os.path.join(image_folder, "fiber")
    tmpRapidPicGrain = os.path.join(image_folder, "grain")

    if os.path.exists(tmpRapidPicFiber):
        pass
    else:
        os.makedirs(tmpRapidPicFiber, exist_ok=True)

    if os.path.exists(tmpRapidPicGrain):
        pass
    else:
        os.makedirs(tmpRapidPicGrain, exist_ok=True)

    with open(os.path.join(output_folder, "image_info_list.csv"), "w") as csvList, open(
        os.path.join(output_folder, "summary_info.csv"), "w"
    ) as csvSum:
        csvList.write(
            "Particle image filename,Size (in pixels),Size (category),Classification (Grain/Fiber)\n"
        )
        video_marker(inputFile)

    if flagDoRapid:
        do_rapid(tmpRapidPicFiber, db_folder)
        do_rapid(tmpRapidPicGrain, db_folder)
