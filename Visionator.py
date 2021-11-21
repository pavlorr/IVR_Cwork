#!/usr/bin/env python3

import math
import cv2
import os
import numpy as np
from statistics import mean


TEST_IMAGE_CAM_1_FILE = os.path.join(os.path.dirname(__file__), 'Test_files/cam_1_baseline.png')
TEST_IMAGE_CAM_1 = cv2.imread(TEST_IMAGE_CAM_1_FILE)
TEST_IMAGE_CAM_2_FILE = os.path.join(os.path.dirname(__file__), 'Test_files/cam_2_baseline.png')
TEST_IMAGE_CAM_2 = cv2.imread(TEST_IMAGE_CAM_2_FILE)
GREEN_TEMPLATE = cv2.imread(os.path.join(os.path.dirname(__file__), 'Test_files/green_template.PNG'), 0)
YELLOW_TEMPLATE = cv2.imread(os.path.join(os.path.dirname(__file__), 'Test_files/yellow_template.PNG'), 0)
RED_TEMPLATE = cv2.imread(os.path.join(os.path.dirname(__file__), 'Test_files/red_template.PNG'), 0)
BLUE_TEMPLATE = cv2.imread(os.path.join(os.path.dirname(__file__), 'Test_files/blue_template.PNG'), 0)
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


def detect_blob(img: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> list:
    """
    Applies mask of given colour to isolate node of that colour
    :param img: image to be masked in BGR
    :type: np.ndarray
    :param lower: lower bound of mask
    :type: np.ndarray
    :param upper: upper bound of mask
    :type: np.ndarray
    :return: the masked image in Black & White
    :rtype: np.ndarray
    """
    mask = cv2.inRange(img, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=3)
    return mask


def get_moments_coords(lower: np.ndarray, upper: np.ndarray) -> list:
    cam1_img = detect_blob(TEST_IMAGE_CAM_1, lower, upper)
    cam2_img = detect_blob(TEST_IMAGE_CAM_2, lower, upper)
    three_d_coords = []
    cam1_mom = cv2.moments(cam1_img)
    cam2_mom = cv2.moments(cam2_img)
    cam1_center = (int(cam1_mom["m10"] / cam1_mom["m00"]), int(cam1_mom["m01"] / cam1_mom["m00"]))
    cam2_center = (int(cam2_mom["m10"] / cam2_mom["m00"]), int(cam2_mom["m01"] / cam2_mom["m00"]))
    three_d_coords.append(cam2_center[0])
    three_d_coords.append(cam1_center[0])
    three_d_coords.append(cam1_center[1])
    return three_d_coords


def get_template_match_coords(lower: np.ndarray, upper: np.ndarray, template: np.ndarray) -> list:
    """
    get the best blob position for the camera with the best visibility
    the best visibility is determined by template matching the blob on the pic
    :param lower: lower value of BGR to be applied as the mask
    :type: np.ndarray
    :param upper: upper value of BGR to be applied as the mask
    :type: np.ndarray
    :param template: template image in greyscale
    :type: np.ndarray
    :return: returns the 3D pixel co-ordinates of the blob
    :rtype: list
    """
    cam1_img = detect_blob(TEST_IMAGE_CAM_1, lower, upper)
    cam2_img = detect_blob(TEST_IMAGE_CAM_2, lower, upper)
    three_d_coords = []
    w, h = template.shape[::-1]
    cam1_tmpl_match = cv2.matchTemplate(cam1_img, template, cv2.TM_CCOEFF)
    cam2_tmpl_match = cv2.matchTemplate(cam2_img, template, cv2.TM_CCOEFF)
    cam1_max_loc = cv2.minMaxLoc(cam1_tmpl_match)[3]
    cam2_max_loc = cv2.minMaxLoc(cam2_tmpl_match)[3]
    three_d_coords.append(cam2_max_loc[0] + w/2)
    three_d_coords.append(cam1_max_loc[0] + w/2)
    three_d_coords.append(cam1_max_loc[1] + h/2)
    # cv2.rectangle(cam1_img, cam1_max_loc, (cam1_max_loc[0] + template.shape[0], cam1_max_loc[1] + template.shape[1]), (0, 0, 0), 2, 8, 0)
    # cv2.rectangle(cam1_tmpl_match, cam1_max_loc, (cam1_max_loc[0] + template.shape[0], cam1_max_loc[1] + template.shape[1]), (0, 0, 0), 2, 8, 0)
    # cv2.imshow('image_window', cam1_img)
    # cv2.imshow('result_window', cam1_tmpl_match)
    # cv2.waitKey(0)
    # cv2.rectangle(cam2_img, cam2_max_loc, (cam2_max_loc[0] + template.shape[0], cam2_max_loc[1] + template.shape[1]), (0,0,0), 2, 8, 0 )
    # cv2.rectangle(cam2_tmpl_match, cam2_max_loc, (cam2_max_loc[0] + template.shape[0], cam2_max_loc[1] + template.shape[1]), (0,0,0), 2, 8, 0 )
    # cv2.imshow('image_window', cam2_img)
    # cv2.imshow('result_window', cam2_tmpl_match)
    # cv2.waitKey(0)
    return three_d_coords


def get_3d_coords(cam1_img_blob: np.ndarray = None, cam2_img_blob: np.ndarray = None) -> list:
    """
    Transforms the 2d pixel co-ordinates into 3d pixel coordinates substituting the missing dimension
    with a placeholder function to be calculated separately
    :param cam1_img_blob: 2d pixel coordinates on y-z axis of the blob center of the investigated colour
    :type: np.ndarray
    :param cam2_img_blob: 2d pixel coordinates on x-z axis of the blob center of the investigated colour
    :type: np.ndarray
    :return: the 3D list representing the 3D pixel coordinates of the center of the blob of the investigated color
    :rtype: list
    raises TypeError: When no parameter is not provided a TypeError is raised
    """
    pass
    # if cam1_img_blob and cam2_img_blob:
    #     return [cam2_img_blob[0], cam1_img_blob[0], mean([cam1_img_blob[1], cam2_img_blob[1]])]
    # elif cam1_img_blob:
    #     return [DUMMY_COORD_VAL, cam1_img_blob[0], cam1_img_blob[1]]
    #     pass
    # elif cam2_img_blob:
    #     return [cam2_img_blob[0], DUMMY_COORD_VAL, cam2_img_blob[1]]
    # raise TypeError('At least one value needs to be supplied')


def calc_angle(in_vect1: np.ndarray, in_vect2: np.ndarray) -> float:
    """
    Angle calculated
    :param in_vect1: 2D vector for axis we are interested in
    :type: np.ndarray
    :param in_vect2: 2D vector for axis we are interested in
    :type: np.ndarray
    :return:
    """
    return np.arccos(in_vect1[0]/np.dot(in_vect1, in_vect1)) - np.arccos(in_vect2[0]/np.dot(in_vect2, in_vect2))


def calc_all_angles(green_3d, yellow_3d, blue_3d, red_3d):
    node_1 = np.array([xi - xj for xi, xj in zip(green_3d, yellow_3d)])
    norm_node_1 = node_1/(np.sqrt(np.sum(node_1**2)))
    node_2 = np.array([xi - xj for xi, xj in zip(yellow_3d, blue_3d)])
    norm_node_2 = node_2/(np.sqrt(np.sum(node_2**2)))
    node_3 = np.array([xi - xj for xi, xj in zip(blue_3d, red_3d)])
    norm_node_3 = node_3/(np.sqrt(np.sum(node_3**2)))

    joint_2_angle_y = calc_angle(np.array([norm_node_1[0], norm_node_1[2]]),
                                 np.array([norm_node_2[0], norm_node_2[2]]))
    joint_3_angle_x = calc_angle(np.array([norm_node_1[1], norm_node_1[2]]),
                                 np.array([norm_node_2[1], norm_node_2[2]]))
    joint_4_angle_y = calc_angle(np.array([norm_node_2[0], norm_node_2[2]]),
                                 np.array([norm_node_3[0], norm_node_3[2]]))

    return[joint_2_angle_y, joint_3_angle_x, joint_4_angle_y]


def calc_all_coords(green_3d_coords: np.ndarray, best_2d_yellow_coords: np.ndarray, best_blue_coords: np.ndarray,
                    best_red_coords: np.ndarray) -> list:
    """
    Placeholder function to check accuracy of pure template matching & review if more accurate approaches need
    to be defined
    """
    pass
    # dummy_val_yel_index = best_2d_yellow_coords.index(DUMMY_COORD_VAL)
    # dummy_val_blue_index = best_2d_yellow_coords.index(DUMMY_COORD_VAL)
    # dummy_val_red_index = best_2d_yellow_coords.index(DUMMY_COORD_VAL)
    # if dummy_val_yel_index == 0:
    #     x = math.sqrt((green_3d_coords[1] - best_2d_yellow_coords[1])**2 +
    #                   (green_3d_coords[2] - best_2d_yellow_coords[2])**2 -
    #                   LINK_1_PIXEL_LENGTH) - green_3d_coords[0]
    #     return [x, best_2d_yellow_coords[1], best_2d_yellow_coords[2]]
    # elif dummy_val_yel_index == 1:
    #     x = math.sqrt((green_3d_coords[0] - best_2d_yellow_coords[0]) ** 2 +
    #                   (green_3d_coords[2] - best_2d_yellow_coords[2]) ** 2 -
    #                   LINK_1_PIXEL_LENGTH) - green_3d_coords[1]
    #     return [best_2d_yellow_coords[0], x, best_2d_yellow_coords[2]]


def get_joint_angles():
    green_3d_coords = get_moments_coords(GREEN_LOWER, GREEN_UPPER)
    yellow_3d_coords = get_template_match_coords(YELLOW_LOWER, YELLOW_UPPER, YELLOW_TEMPLATE)
    blue_3d_coords = get_template_match_coords(BLUE_LOWER, BLUE_UPPER, BLUE_TEMPLATE)
    red_3d_coords = get_template_match_coords(RED_LOWER, RED_UPPER, RED_TEMPLATE)
    calc_all_angles(green_3d_coords, yellow_3d_coords, blue_3d_coords, red_3d_coords)


if __name__ == '__main__':
    get_joint_angles()
