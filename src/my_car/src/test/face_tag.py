#!/usr/bin/env python3
import rospy
import tf2_ros
from tf.transformations import quaternion_multiply, quaternion_matrix, quaternion_from_euler
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, PoseStamped 
import time
import math

rospy.init_node('face_tag')

x_ofs = 0.7     # m
y_ofs = 0.5

# Info of target tag in Map 
tag_position = [0.745, -4.69, 0.0]      # x, y, z
tag_orientation = [0.0, 0.0, 0.9170601, 0.3987491]      # x, y, z, w 

# Real Tag info
tag_pose = TransformStamped()
tag_pose.header.stamp = rospy.Time.now()
tag_pose.header.frame_id = "map"
tag_pose.child_frame_id = "tag_0"
tag_pose.transform.translation.x = tag_position[0]
tag_pose.transform.translation.y = tag_position[1]
tag_pose.transform.translation.z = tag_position[2]
tag_pose.transform.rotation.x = tag_orientation[0]
tag_pose.transform.rotation.y = tag_orientation[1]
tag_pose.transform.rotation.z = tag_orientation[2]
tag_pose.transform.rotation.w = tag_orientation[3]

# Transfer from real tag
rot_matrix = quaternion_matrix(tag_orientation)
position_offset = rot_matrix.dot([x_ofs, y_ofs, 0.0, 1.0])
q = quaternion_from_euler(0, 0, math.radians(180))
dummy_orientation = quaternion_multiply(tag_orientation, q)

# Dummy Tag info
dummy_tag = TransformStamped()
dummy_tag.header.stamp = rospy.Time.now()
dummy_tag.header.frame_id = "map"
dummy_tag.child_frame_id = "dummy_tag"
dummy_tag.transform.translation.x = tag_position[0] + position_offset[0]
dummy_tag.transform.translation.y = tag_position[1] + position_offset[1]
dummy_tag.transform.translation.z = tag_position[2] + position_offset[2]
dummy_tag.transform.rotation.x = dummy_orientation[0]
dummy_tag.transform.rotation.y = dummy_orientation[1]
dummy_tag.transform.rotation.z = dummy_orientation[2]
dummy_tag.transform.rotation.w = dummy_orientation[3]

# Goal
goal_info = PoseStamped()
goal_info.header.stamp = rospy.Time.now()
goal_info.header.frame_id = "map"
goal_info.pose.position.x = tag_position[0] + position_offset[0]
goal_info.pose.position.y = tag_position[1] + position_offset[1]
goal_info.pose.position.z = tag_position[2] + position_offset[2]
goal_info.pose.orientation.x = dummy_orientation[0]
goal_info.pose.orientation.y = dummy_orientation[1]
goal_info.pose.orientation.z = dummy_orientation[2]
goal_info.pose.orientation.w = dummy_orientation[3]



if __name__ == '__main__':
    '''
    tfbcaster = tf2_ros.StaticTransformBroadcaster()
    tfbcaster.sendTransform([tag_pose, dummy_tag])
    rospy.spin()
    '''

    # set the Pubkisher object               
    pub = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=10)
    rospy.sleep(0.5)
    pub.publish(goal_info)
    
    




