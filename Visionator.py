#!/usr/bin/env python3

import cv2
import os
import numpy as np

TEST_IMAGE_CAM_1_FILE = os.path.join(os.path.dirname(__file__), 'Test_files/cam_2_invest.png')
TEST_IMAGE_CAM_1 = cv2.imread(TEST_IMAGE_CAM_1_FILE)
TEST_IMAGE_CAM_2_FILE = os.path.join(os.path.dirname(__file__), 'Test_files/cam_2_baseline.png')
TEST_IMAGE_CAM_2 = cv2.imread(TEST_IMAGE_CAM_2_FILE)
YELLOW_TEMPLATE = cv2.imread(os.path.join(os.path.dirname(__file__), 'Test_files/yellow_template.PNG'), 0)
RED_TEMPLATE = cv2.imread(os.path.join(os.path.dirname(__file__), 'Test_files/red_template.PNG'), 0)
BLUE_TEMPLATE = cv2.imread(os.path.join(os.path.dirname(__file__), 'Test_files/blue_template.PNG'), 0)
GREEN_TEMPLATE = cv2.imread(os.path.join(os.path.dirname(__file__), 'Test_files/green_template.PNG'), 0)
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
    return mask


def get_best(lower: np.ndarray, upper: np.ndarray, template: np.ndarray):
    cam1_yellow = detect_blob(TEST_IMAGE_CAM_1, lower, upper)
    cam_2_yellow = detect_blob(TEST_IMAGE_CAM_2, lower, upper)
    w, h = template.shape[::-1]
    cam_1_tmpl_mtch = cv2.matchTemplate(cam1_yellow, template, cv2.TM_CCORR)
    cam_2_tmpl_mtch = cv2.matchTemplate(cam_2_yellow, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cam_1_tmpl_mtch)
    min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(cam_2_tmpl_mtch)
    if max_val > max_val2:
        center_of_blob = [max_loc[0] + w/2, max_loc[1] + h/2]
    else:
        center_of_blob = [max_loc2[0] + w/2, max_loc2[1] + h/2]
    return center_of_blob


def calc_angle():
    pass


def cam_1_coord() -> np.ndarray:
    pass


def cam_2_coord() -> np.ndarray:
    pass


def calc_3d_coord() -> np.ndarray:
    pass


def main():
    best_blue_coords = get_best(BLUE_LOWER, BLUE_UPPER, BLUE_TEMPLATE)
    best_red_coords = get_best(RED_LOWER, RED_UPPER, RED_TEMPLATE)
    best_green_coords = get_best(GREEN_LOWER, GREEN_UPPER, GREEN_TEMPLATE)
    best_yellow_coords = get_best(YELLOW_LOWER, YELLOW_UPPER, YELLOW_TEMPLATE)


if __name__ == '__main__':
    main()
