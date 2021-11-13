#!/usr/bin/env python3

import rospy
import math
from std_msgs.msg import Float64


# Global Variables
JOINT_1_TOPIC = '/robot/joint1_position_controller/command'
JOINT_2_TOPIC = '/robot/joint2_position_controller/command'
JOINT_3_TOPIC = '/robot/joint3_position_controller/command'
JOINT_4_TOPIC = '/robot/joint4_position_controller/command'


class JointHandler:
    """
    Class to control Joints & calculation
    Attributes
    ----------
    pkg : Float64()
        controls the return value
    t : Float64
        time used to calculate joint movements

    Methods
    -------
    joint1 : Float64()
        updates the pkg attribute with the movement associated to joint 1 and returns it - currently fixed
    joint2 : Float64()
        updates the pkg attribute with the movement associated to joint 2 and returns it
    joint3 : Float64()
        updates the pkg attribute with the movement associated to joint 3 and returns it
    joint4 : Float64()
        updates the pkg attribute with the movement associated to joint 4 and returns it
    """
    def __init__(self, t: Float64):
        self.pkg = Float64()
        self.t = t

    def joint1(self) -> Float64:
        # joint 1 not used for Q1
        pass

    def joint2(self) -> Float64:
        self.pkg.data = (math.pi/2) * math.sin(math.pi*self.t/15)
        return self.pkg

    def joint3(self) -> Float64:
        self.pkg.data = (math.pi/2) * math.sin(math.pi*self.t/20)
        return self.pkg

    def joint4(self) -> Float64:
        self.pkg.data = (math.pi/2) * math.sin(math.pi*self.t/18)
        return self.pkg


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
        self.rate = rospy.Rate(0.5)

    def callback(self):
        while not rospy.is_shutdown():
            for i in range(30):
                jh = JointHandler(i/2)
                # joint 1 is fixed, so no publishing
                # a better way of publishing can be found but does not really matter
                #self.publicator_joint1.publish(JointHandler().pkg)
                self.publicator_joint2.publish(jh.joint2())
                self.publicator_joint3.publish(jh.joint3())
                self.publicator_joint4.publish(jh.joint4())
                self.rate.sleep()


# run the code if the node is called
if __name__ == '__main__':
    try:
        p = RadianPublicator()
        p.callback()
    except rospy.ROSInterruptException:
        pass
