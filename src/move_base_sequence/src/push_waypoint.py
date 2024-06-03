#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped 


rospy.init_node("send_goal")

goal_publisher = rospy.Publisher("/move_base_sequence/corner_pose", PoseStamped, queue_size=5)

rospy.sleep(1)

path = [[0.9900000095367432, 2.2799994945526123, -0.9999134765952385, 0.013154441197683974],
        [-0.9299993515014648, 2.1999995708465576, -0.7318633349163792, 0.6814514355440714],
        [-0.8699997067451477, 1.9399996995925903, 0.015619180184782262, 0.9998780131647836],
        [0.9500000476837158, 2.059999465942383, -0.7071067966408575, 0.7071067657322372],
        [0.9300000071525574, 1.7199994325637817, -0.9998407036600967, 0.017848453840112192],
        [-1.5099996328353882, 1.4399999380111694, -0.38268344246110436, 0.9238795283293805],
        [1.809999704360962, -0.2800004184246063, 0.7180650077396533, 0.6959760374178494]]


goal = PoseStamped()

for i in range(len(path)):
    goal.header.seq = i
    goal.header.stamp = rospy.Time.now()
    goal.header.frame_id = "map"
    
    goal.pose.position.x = path[i][0]
    goal.pose.position.y = path[i][1]
    goal.pose.position.z = 0.0

    goal.pose.orientation.x = 0.0
    goal.pose.orientation.y = 0.0
    goal.pose.orientation.z = path[i][2]
    goal.pose.orientation.w = path[i][3]
    
    rospy.sleep(0.05)

    goal_publisher.publish(goal)







