#!/usr/bin/env python3

import cv2
import os
import numpy as np

TEST_IMAGE_CAM_1_FILE = os.path.join(os.path.dirname(__file__), 'Test_files/cam_1.png')
TEST_IMAGE_CAM_2_FILE = os.path.join(os.path.dirname(__file__), 'Test_files/cam_2.png')
YELLOW_TEMPLATE = cv2.imread(os.path.join(os.path.dirname(__file__), 'Test_files/yellow_template.PNG'), 0)
LINK_1_LENGTH = 4.0
LINK_1_PIXEL_LENGTH = 105
LINK_2_LENGTH = 0.0
LINK_3_LENGTH = 3.2
LINK_3_PIXEL_LENGTH = 80
LINK_4_LENGTH = 2.8
LINK_4_PIXEL_LENGTH = 76
# blobs are darker so the upper limit needs to account for that
BLUE_LOWER = np.array([100, 0, 0], np.uint8)
BLUE_UPPER = np.array([255, 30, 30], np.uint8)
RED_LOWER = np.array([0, 0, 100], np.uint8)
RED_UPPER = np.array([20, 20, 255], np.uint8)
YELLOW_LOWER = np.array([0, 100, 100], np.uint8)
YELLOW_UPPER = np.array([20, 255, 255], np.uint8)
GREEN_LOWER = np.array([0, 100, 0], np.uint8)
GREEN_UPPER = np.array([20, 255, 20], np.uint8)
CAM2_Z_CORR_VAL = -1  # pixel correction for z axis measurements for cam 2


def detect_blob(img: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    mask = cv2.inRange(img, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=3)
    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array([cx, cy])


def helper(img, lower, upper):
    mask = cv2.inRange(img, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)


def pixel_to_meter(center_1, center_2, length):
    dist = np.sum((center_1 - center_2) ** 2)
    return length / np.sqrt(dist)


def calc_angle():
    pass


def cam_1_coord() -> np.ndarray:
    pass


def cam_2_coord() -> np.ndarray:
    pass


def calc_3d_coord() -> np.ndarray:
    pass


# def template_match(img: np.ndarray, lower: np.ndarray, upper: np.ndarray, template: np.ndarray = None) -> np.ndarray:
#     blob_centre = detect_blob(img, lower, upper)
#     mask = cv2.inRange(img, lower, upper)
#     res = cv2.matchTemplate(mask, template, 0)
#     print(template.shape)
#     exit()
#     ROI = mask[int(blob_centre[1] - template.shape[0] / 2): int(blob_centre[1] + template.shape[0] / 2) + 1,
#           int(blob_centre[0] - template.shape[1] / 2): int(blob_centre[0] + template.shape[1] / 2) + 1]
#     ROI = ROI[0:template.shape[0], 0:template.shape[1]]
#     dist = cv2.distanceTransform(cv2.bitwise_not(ROI), cv2.DIST_L2, 0)
#     res = dist*template[0:dist.shape[0], 0:dist.shape[1]]
#     pass


def main():
    cam_1_img = cv2.imread(TEST_IMAGE_CAM_1_FILE, 1)
    cam_2_img = cv2.imread(TEST_IMAGE_CAM_2_FILE, 1)
    cam1_green_pix = detect_blob(cam_1_img, GREEN_LOWER, GREEN_UPPER)  # [0,0] point for cam 1
    cam2_green_pix = detect_blob(cam_2_img, GREEN_LOWER, GREEN_UPPER)  # [0,0] point for cam 2
    cam1_yellow_pix = detect_blob(cam_1_img, YELLOW_LOWER, YELLOW_UPPER)
    cam2_yellow_pix = detect_blob(cam_2_img, YELLOW_LOWER, YELLOW_UPPER)
    #template_match(cam_1_img, YELLOW_LOWER, YELLOW_UPPER, YELLOW_TEMPLATE)
    best_blue_coords = None
    best_red_coords = None
    best_green_coords = None
    best_yellow_coords = None


if __name__ == '__main__':
    main()
