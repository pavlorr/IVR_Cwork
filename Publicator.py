#!/usr/bin/env python3

import rospy
import math
import sys
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
    @todo: Would like to automate the publication & subscription but not possible in the time remaining
    :param joint_no: Number of joint to be rotated
    :type: int
    :param time: time in seconds given to base the rotation on
    :type: float
    :return: the type of variable & angle of rotation for the given axis & time
    :rtype: std_msgs.msg.Float64
    """
    pkg = Float64()
    if joint_no == 1:
        pass
    elif joint_no == 2:
        pkg.data = (math.pi/2) * math.sin(math.pi*time/15)
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
        #self.publicator_joint1 = rospy.Publisher(JOINT_1_TOPIC, Float64, queue_size=1)
        self.publicator_joint2 = rospy.Publisher(JOINT_2_TOPIC, Float64, queue_size=1)
        self.publicator_joint3 = rospy.Publisher(JOINT_3_TOPIC, Float64, queue_size=1)
        self.publicator_joint4 = rospy.Publisher(JOINT_4_TOPIC, Float64, queue_size=1)

    def callback(self, seconds: float):
        if not rospy.is_shutdown():
            # joint 1 is fixed, so no publishing
            # a better way of publishing can be found but does not really matter
            #self.publicator_joint1.publish(JointHandler().pkg)
            self.publicator_joint2.publish(joint_angle(JOINT_2, seconds))
            self.publicator_joint3.publish(joint_angle(JOINT_3, seconds))
            self.publicator_joint4.publish(joint_angle(JOINT_4, seconds))


# run the code if the node is called
if __name__ == '__main__':
    try:
        p = RadianPublicator()
        p.callback(float(sys.argv[1]))
    except rospy.ROSInterruptException:
        pass
