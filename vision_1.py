#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import os
import math
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError

GREEN_TEMPLATE = cv2.imread(os.path.join(os.path.dirname(__file__), 'Test_files/green_template.PNG'), 0)
YELLOW_TEMPLATE = cv2.imread(os.path.join(os.path.dirname(__file__), 'Test_files/yellow_template.PNG'), 0)
RED_TEMPLATE = cv2.imread(os.path.join(os.path.dirname(__file__), 'Test_files/red_template.PNG'), 0)
BLUE_TEMPLATE = cv2.imread(os.path.join(os.path.dirname(__file__), 'Test_files/blue_template.PNG'), 0)
# blobs are darker so the upper limit needs to account for that
LINK_1_LENGTH = 4.0
LINK_1_PIXEL_LENGTH = 105
LINK_2_LENGTH = 0.0
LINK_3_LENGTH = 3.2
LINK_3_PIXEL_LENGTH = 80
LINK_4_LENGTH = 2.8
LINK_4_PIXEL_LENGTH = 76
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


def get_moments_coords(img1: np.ndarray, img2: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> list:
    """
    Get green's joint base co-ordinates from moments instead of template. Green is always visible so moments can be used
    :param img1: image from camera 1
    :type: np.ndarray
    :param img2: image from camera 2
    :type: np.ndarray
    :param lower: lower bound of BGR mask needed to be applied
    :type: np.ndarray
    :param upper: upper bound of BGR mask needed to be applied
    :return: 3D coordinates of center
    :rtype: list
    """
    cam1_img = detect_blob(img1, lower, upper)
    cam2_img = detect_blob(img2, lower, upper)
    three_d_coords = []
    cam1_mom = cv2.moments(cam1_img)
    cam2_mom = cv2.moments(cam2_img)
    cam1_center = (int(cam1_mom["m10"] / cam1_mom["m00"]), int(cam1_mom["m01"] / cam1_mom["m00"]))
    cam2_center = (int(cam2_mom["m10"] / cam2_mom["m00"]), int(cam2_mom["m01"] / cam2_mom["m00"]))
    three_d_coords.append(cam2_center[0])
    three_d_coords.append(cam1_center[0])
    three_d_coords.append(cam1_center[1])
    return three_d_coords


def get_template_match_coords(img1: np.ndarray, img2: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                              template: np.ndarray) -> list:
    """
    get the best blob position for the camera with the best visibility
    the best visibility is determined by template matching the blob on the pic
    :param img1: image from camera 1
    :type: np.ndarray
    :param img2: image from camera 2
    :type: np.ndarray
    :param lower: lower value of BGR to be applied as the mask
    :type: np.ndarray
    :param upper: upper value of BGR to be applied as the mask
    :type: np.ndarray
    :param template: template image in greyscale
    :type: np.ndarray
    :return: returns the 3D pixel co-ordinates of the blob
    :rtype: list
    """
    cam1_img = detect_blob(img1, lower, upper)
    cam2_img = detect_blob(img2, lower, upper)
    three_d_coords = []
    w, h = template.shape[::-1]
    cam1_tmpl_match = cv2.matchTemplate(cam1_img, template, cv2.TM_CCOEFF)
    cam2_tmpl_match = cv2.matchTemplate(cam2_img, template, cv2.TM_CCOEFF)
    cam1_max_loc = cv2.minMaxLoc(cam1_tmpl_match)[3]
    cam2_max_loc = cv2.minMaxLoc(cam2_tmpl_match)[3]
    three_d_coords.append(cam2_max_loc[0] + w/2)
    three_d_coords.append(cam1_max_loc[0] + w/2)
    three_d_coords.append(cam1_max_loc[1] + h/2)
    return three_d_coords


def calc_angle(in_vect1: np.ndarray, in_vect2: np.ndarray) -> float:
    """
    Angle calculated
    :param in_vect1: 2D vector for axis we are interested in
    :type: np.ndarray
    :param in_vect2: 2D vector for axis we are interested in
    :type: np.ndarray
    :return:
    """
    pass


def calc_all_angles(green_3d, yellow_3d, blue_3d, red_3d):
    node_2 = np.array([xi - xj for xi, xj in zip(yellow_3d, blue_3d)])
    node_3 = np.array([xi - xj for xi, xj in zip(blue_3d, red_3d)])
    norm_node_2 = node_2 / math.sqrt(np.sum(node_2 ** 2))
    norm_node_3 = node_3 / math.sqrt(np.sum(node_3 ** 2))
    joint_2_angle_y = -np.arcsin(norm_node_2[0])
    joint_3_angle_x = np.arcsin(norm_node_2[1])
    joint_4_sign_help = np.cross(norm_node_2, norm_node_3)
    joint_4_angle_y_abs = np.arccos(np.dot(norm_node_2, norm_node_3))
    joint_4_angle_y = joint_4_angle_y_abs if joint_4_sign_help[0] < 0 else -1*joint_4_angle_y_abs
    print([joint_2_angle_y, joint_3_angle_x, joint_4_angle_y])
    return [joint_2_angle_y, joint_3_angle_x, joint_4_angle_y]


def get_joint_angles(img: np.ndarray, img2: np.ndarray) -> list:
    green_3d_coords = get_moments_coords(img, img2, GREEN_LOWER, GREEN_UPPER)
    yellow_3d_coords = get_template_match_coords(img, img2, YELLOW_LOWER, YELLOW_UPPER, YELLOW_TEMPLATE)
    blue_3d_coords = get_template_match_coords(img, img2, BLUE_LOWER, BLUE_UPPER, BLUE_TEMPLATE)
    red_3d_coords = get_template_match_coords(img, img2, RED_LOWER, RED_UPPER, RED_TEMPLATE)
    return calc_all_angles(green_3d_coords, yellow_3d_coords, blue_3d_coords, red_3d_coords)


class ImageProcessing:
    def __init__(self):
        rospy.init_node('image_processing', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.callback1)
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)
        self.joint_2_angle = rospy.Publisher("Joint_2_angle", Float64, queue_size=1)
        self.joint_3_angle = rospy.Publisher("Joint_3_angle", Float64, queue_size=1)
        self.joint_4_angle = rospy.Publisher("Joint_4_angle", Float64, queue_size=1)
        self.rate = 0.5

    def callback2(self, data):
        self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        jointanglelist = get_joint_angles(self.cv_image1, self.cv_image2)
        joint_angle = Float64()
        joint_angle.data = jointanglelist[0]
        self.joint_2_angle.publish(joint_angle)
        joint_angle.data = jointanglelist[1]
        self.joint_3_angle.publish(joint_angle)
        joint_angle.data = jointanglelist[2]
        self.joint_4_angle.publish(joint_angle)

    def callback1(self, data):
        self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")


def main():
    ic = ImageProcessing()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
