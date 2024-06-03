#!/usr/bin/env python3
import rospy
from enum import Enum
import tf2_ros
from tf import TransformBroadcaster
from tf.transformations import quaternion_multiply, quaternion_from_euler, euler_from_quaternion, quaternion_matrix
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseActionResult
from math import atan2

class Alignment():
    def __init__(self):
        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.tfbuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfbuffer)
        self.tfbcaster = TransformBroadcaster()
        self.rate = rospy.Rate(10.0)
        self.vel = Twist()
        self.AlignmentSequence = Enum('AlignmentSequence', 'wait_reach_goal searching_code make_target_frame changing_direction moving_nearby facing_code finished')
        self.current_sequence = self.AlignmentSequence.wait_reach_goal.value
        self.target_angle = 0.0
        self.turn_reach = 0
        self.first = True
    
    def main(self):
        ####################
        ## Searching code ##
        ####################
        if self.current_sequence is self.AlignmentSequence.wait_reach_goal.value:
            rospy.wait_for_message("move_base/result", MoveBaseActionResult)
            print("Start !")
            self.current_sequence = self.AlignmentSequence.searching_code.value

   


#========================================================================================

if __name__ == '__main__':
    rospy.init_node('alignment')
    alignment = Alignment()
    while not rospy.is_shutdown():
        alignment.main()
        rospy.sleep(0.1)
    





