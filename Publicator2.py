#!/usr/bin/env python3

import rospy
import math
from forward_kinematics import forward_kinematics
from std_msgs.msg import Float64


# Global Variables
JOINT_1_TOPIC = '/robot/joint1_position_controller/command'
JOINT_2_TOPIC = '/robot/joint2_position_controller/command'
JOINT_3_TOPIC = '/robot/joint3_position_controller/command'
JOINT_4_TOPIC = '/robot/joint4_position_controller/command'
JOINT_1 = 1
JOINT_2 = 2
JOINT_3 = 3
JOINT_4 = 4


def joint_angle(joint_no: int, time: float) -> Float64:
    """
    Calculates angle of rotation for each joint based on exercise description & given time
    :param joint_no: Number of joint to be rotated
    :type: int
    :param time: time in seconds given to base the rotation on
    :type: float
    :return: the type of variable & angle of rotation for the given axis & time
    :rtype: std_msgs.msg.Float64
    """
    pkg = Float64()
    if joint_no == 1:
        pkg.data = math.pi * math.sin(math.pi*time/28)
    elif joint_no == 2:
        pass
    elif joint_no == 3:
        pkg.data = (math.pi/2) * math.sin(math.pi*time/20)
    elif joint_no == 4:
        pkg.data = (math.pi/2) * math.sin(math.pi*time/18)
    else:
        raise TypeError('Not supported joint_no: ' + str(joint_no))
    return pkg


class RadianPublicator:
    """
    Class to control the publishing of the joint movements
    Attributes
    ----------
    publicator_joint1 : rospy.Publisher
        creates the publisher to the associated ROS standard topic for the movement of the robot joints.
        Currently commented out for joint1
    publicator_joint2 : rospy.Publisher
        creates the publisher to the associated ROS standard topic for the movement of the robot joints
    publicator_joint3 : rospy.Publisher
        creates the publisher to the associated ROS standard topic for the movement of the robot joints
    publicator_joint4 : rospy.Publisher
        creates the publisher to the associated ROS standard topic for the movement of the robot joints
    rate : Float64
        the rate (Hz) of publications to the topics

    Methods
    -------
    callback
        publishes the relevant angles to the associated ROS topics for each robot joint
        waits for the prescribed rate until it publishes again
    """
    def __init__(self):
        # initialize the node named Publicator that will publish the movements of the robot
        rospy.init_node('Publicator', anonymous=True)
        # initialize the 4 publishers - 1 for each joint
        # joint 1 is commented out for the first question as it's assumed fixed
        self.publicator_joint1 = rospy.Publisher(JOINT_1_TOPIC, Float64, queue_size=1)
        #self.publicator_joint2 = rospy.Publisher(JOINT_2_TOPIC, Float64, queue_size=1)
        self.publicator_joint3 = rospy.Publisher(JOINT_3_TOPIC, Float64, queue_size=1)
        self.publicator_joint4 = rospy.Publisher(JOINT_4_TOPIC, Float64, queue_size=1)
        self.rate = rospy.Rate(0.1)
        self.target_pos_sub = rospy.Subscriber('/target_pos', Float64MultiArray, self.callback_traj)



    def callback(self):
        seconds = 0
        while not rospy.is_shutdown():
            print(joint_angle(JOINT_1, seconds), joint_angle(JOINT_3, seconds), joint_angle(JOINT_4, seconds))
            # joint 1 is fixed, so no publishing
            self.publicator_joint1.publish(joint_angle(JOINT_1, seconds))
            #elf.publicator_joint2.publish(joint_angle(JOINT_2, seconds))
            self.publicator_joint3.publish(joint_angle(JOINT_3, seconds))
            self.publicator_joint4.publish(joint_angle(JOINT_4, seconds))
            print(forward_kinematics(joint_angle(JOINT_1, seconds), joint_angle(JOINT_3, seconds),
                                     joint_angle(JOINT_4, seconds)))
            self.rate.sleep()
            seconds += 1


# run the code if the node is called
if __name__ == '__main__':
    try:
        p = RadianPublicator()
        p.callback()
    except rospy.ROSInterruptException:
        pass
