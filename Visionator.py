#!/usr/bin/env python3

import cv2
import os
import numpy as np

TEST_IMAGE_CAM_1_FILE = os.path.join(os.path.dirname(__file__), 'Test_files/cam_1.png')
TEST_IMAGE_CAM_2_FILE = os.path.join(os.path.dirname(__file__), 'Test_files/cam_2.png')
LINK_1_LENGTH = 4.0
LINK_2_LENGTH = 0.0
LINK_3_LENGTH = 3.2
LINK_4_LENGTH = 2.8


def detect_blue(img: np.ndarray):
    blue_lower = np.array([100, 0, 0], np.uint8)
    blue_upper = np.array([255, 0, 0], np.uint8)
    mask = cv2.inRange(img, blue_lower, blue_upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array([cx, cy])


def detect_red(img: np.ndarray):
    mask = cv2.inRange(img, (0, 0, 100), (0, 0, 255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array([cx, cy])


def detect_yellow(img: np.ndarray):
    mask = cv2.inRange(img, (0, 100, 100), (0, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array([cx, cy])


def detect_green(img: np.ndarray):
    mask = cv2.inRange(img, (0, 100, 0), (0, 255, 0))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array([cx, cy])


def pixel_to_meter(center_1, center_2, length):
    dist = np.sum((center_1 - center_2) ** 2)
    return length / np.sqrt(dist)


def calc_angle():
    pass


def main():
    cam_1_img = cv2.imread(TEST_IMAGE_CAM_1_FILE, 1)
    cam_2_img = cv2.imread(TEST_IMAGE_CAM_2_FILE, 1)


if __name__ == '__main__':
    main()
