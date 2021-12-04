#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import os
import math
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError


JOINT_1_TOPIC = '/robot/joint1_position_controller/command'
JOINT_3_TOPIC = '/robot/joint3_position_controller/command'
JOINT_4_TOPIC = '/robot/joint4_position_controller/command'
TARGET_POS_TOPIC = '/target_pos'
ALL_ANGLES_TOPIC = '/all_joints_angle'
DIF_TOPIC = '/pos_dif_tracker'
TARGET_POS_X_TOPIC = '/target_pos_x'
TARGET_POS_Y_TOPIC = '/target_pos_y'
TARGET_POS_Z_TOPIC = '/target_pos_z'
EFFECTOR_POS_X_TOPIC = '/effector_pos_x'
EFFECTOR_POS_Y_TOPIC = '/effector_pos_y'
EFFECTOR_POS_Z_TOPIC = '/effector_pos_z'
# blobs are darker so the upper limit needs to account for that
LINK_1_LENGTH = 4.0
LINK_1_PIXEL_LENGTH = 105
LINK_3_PIXEL_LENGTH = 80
LINK_4_PIXEL_LENGTH = 76
# use this to go from meter coordinates to pixel coordinates
METER_TO_PIXEL = LINK_1_PIXEL_LENGTH/LINK_1_LENGTH


class RobotControl:
    def __init__(self):
        rospy.init_node('robot_control', anonymous=True)
        self.bridge = CvBridge()
        self.t_previous = rospy.get_time()
        self.all_angles = []
        self.all_joint_angles_sub = rospy.Subscriber(ALL_ANGLES_TOPIC, Float64MultiArray, self.callback_joint_angles)
        self.target_pos_sub = rospy.Subscriber(TARGET_POS_TOPIC, Float64MultiArray, self.callback_move)
        self.publicator_joint1 = rospy.Publisher(JOINT_1_TOPIC, Float64, queue_size=1)
        self.publicator_joint3 = rospy.Publisher(JOINT_3_TOPIC, Float64, queue_size=1)
        self.publicator_joint4 = rospy.Publisher(JOINT_4_TOPIC, Float64, queue_size=1)
        self.ef_x = rospy.Publisher(EFFECTOR_POS_X_TOPIC, Float64, queue_size=1)
        self.ef_y = rospy.Publisher(EFFECTOR_POS_Y_TOPIC, Float64, queue_size=1)
        self.ef_z = rospy.Publisher(EFFECTOR_POS_Z_TOPIC, Float64, queue_size=1)
        self.target_x = rospy.Publisher(TARGET_POS_X_TOPIC, Float64, queue_size=1)
        self.target_y = rospy.Publisher(TARGET_POS_Y_TOPIC, Float64, queue_size=1)
        self.target_z = rospy.Publisher(TARGET_POS_Z_TOPIC, Float64, queue_size=1)

    def callback_joint_angles(self, data):
        self.all_angles = data.data
        self.end_effector = self.forward_kinematics(self.all_angles)
        self.jacobian = self.jacobian_matrix(self.all_angles)
        pkg = Float64()
        if self.target_pos is not None:
            pkg.data = self.end_effector[0]
            self.ef_x.publish(pkg)
            pkg.data = self.end_effector[1]
            self.ef_y.publish(pkg)
            pkg.data = self.end_effector[2]
            self.ef_z.publish(pkg)
            pkg.data = self.target_pos[0]
            self.target_x.publish(pkg)
            pkg.data = self.target_pos[1]
            self.target_y.publish(pkg)
            pkg.data = self.target_pos[2]
            self.target_z.publish(pkg)

    def callback_move(self, data):
        if not self.all_angles:
            return
        self.target_pos = np.array(data.data)*METER_TO_PIXEL
        inverse_jacobian = np.linalg.pinv(self.jacobian)
        current_time = rospy.get_time()
        dt = current_time - self.t_previous
        self.t_previous = current_time
        self.pos_diff = (self.target_pos - self.end_effector) / dt
        target_angles = self.all_angles + (dt * np.dot(inverse_jacobian, self.pos_diff.transpose()))
        call_angle = Float64()
        call_angle.data = target_angles[0]
        self.publicator_joint1.publish(call_angle)
        call_angle.data = target_angles[1]
        self.publicator_joint3.publish(call_angle)
        call_angle.data = target_angles[2]
        self.publicator_joint4.publish(call_angle)

    def forward_kinematics(self, all_angles) -> np.array:
        r_array = np.array([0, 0, 0])
        sin1 = np.sin(all_angles[0])
        sin3 = np.sin(all_angles[1])
        sin4 = np.sin(all_angles[2])
        cos1 = np.cos(all_angles[0])
        cos3 = np.cos(all_angles[1])
        cos4 = np.cos(all_angles[2])
        r_array[0] = LINK_4_PIXEL_LENGTH * cos1 * sin4 +\
                     sin3 * sin1 * (LINK_3_PIXEL_LENGTH + LINK_4_PIXEL_LENGTH * cos4)
        r_array[1] = LINK_4_PIXEL_LENGTH * sin4 * sin1 - \
                     cos1 * sin3 * (LINK_3_PIXEL_LENGTH + LINK_4_PIXEL_LENGTH * cos4)
        r_array[2] = LINK_1_PIXEL_LENGTH + cos3 * (LINK_3_PIXEL_LENGTH + LINK_4_PIXEL_LENGTH * cos4)
        return r_array

    def jacobian_matrix(self, all_angles) -> np.array:
        sin1 = np.sin(all_angles[0])
        sin3 = np.sin(all_angles[1])
        sin4 = np.sin(all_angles[2])
        cos1 = np.cos(all_angles[0])
        cos3 = np.cos(all_angles[1])
        cos4 = np.cos(all_angles[2])
        jacobian = np.zeros((3, 3))
        jacobian[0][0] = -LINK_4_PIXEL_LENGTH*sin1*sin4 + cos1*sin3*(LINK_3_PIXEL_LENGTH + LINK_4_PIXEL_LENGTH*cos4)
        jacobian[1][0] = LINK_4_PIXEL_LENGTH*cos1*sin4 + sin1*sin3*(LINK_3_PIXEL_LENGTH + LINK_4_PIXEL_LENGTH*cos4)
        jacobian[2][0] = 0
        jacobian[0][1] = sin1*cos3*(LINK_3_PIXEL_LENGTH + LINK_4_PIXEL_LENGTH*cos4)
        jacobian[1][1] = -cos1 * cos3 * (LINK_3_PIXEL_LENGTH + LINK_4_PIXEL_LENGTH * cos4)
        jacobian[2][1] = -sin3*(LINK_3_PIXEL_LENGTH + LINK_4_PIXEL_LENGTH*cos4)
        jacobian[0][2] = LINK_4_PIXEL_LENGTH*cos1*cos4 - LINK_4_PIXEL_LENGTH*sin1*sin3*sin4
        jacobian[1][2] = LINK_4_PIXEL_LENGTH*sin1*cos4 + LINK_4_PIXEL_LENGTH*sin1*sin3*sin4
        jacobian[2][2] = -LINK_4_PIXEL_LENGTH*sin4*cos3
        return jacobian


def main():
    ic = RobotControl()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
