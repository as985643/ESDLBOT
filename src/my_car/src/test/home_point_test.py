#!/usr/bin/env python3
import rospy
import tf2_ros
from tf.transformations import quaternion_multiply, quaternion_matrix
from geometry_msgs.msg import PoseWithCovarianceStamped
import time


rospy.init_node('initialpose_pub')

# Set rotation angle
#angle = 0
def listen_base_footprint_to_tag(tag_name):
    '''
    Listening until get the specify tf frame, then caculate the relative pose
    '''
    tfbuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfbuffer)
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            trans = tfbuffer.lookup_transform('base_footprint', tag_name, rospy.Time())
            orientation = trans.transform.rotation
            break
        except:
            continue
        rate.sleep()
    return trans
        
        
def talker():
    
    # get the pose from base_footprint to tag_0
    robot_trans = listen_base_footprint_to_tag("tag_0")
    # rotation [x, y, z, w].T
    robot_quat = [robot_trans.transform.rotation.x,
                  robot_trans.transform.rotation.y,
                  robot_trans.transform.rotation.z,
                 -robot_trans.transform.rotation.w]
    # position [x, y, z, 1].T           
    robot_dist = [robot_trans.transform.translation.x,
                  robot_trans.transform.translation.y, 
                  robot_trans.transform.translation.z,
                  1]
    
    # set the Pubkisher object               
    pub = rospy.Publisher('initialpose', PoseWithCovarianceStamped, queue_size=10)
    
    # Set home pose    
    home = PoseWithCovarianceStamped()
    home.header.stamp = rospy.Time.now()
    home.header.frame_id = "map"  
    home.pose.pose.position.x = 2.73	#0.376
    home.pose.pose.position.y = 2.71	#0.109
    home.pose.pose.orientation.x = 0.0	#0.5	
    home.pose.pose.orientation.y = 0.7071068	#-0.5	
    home.pose.pose.orientation.z = 0.7071068	#-0.5	
    home.pose.pose.orientation.w = 0.0	#0.5	   
    home_quat = [home.pose.pose.orientation.x, home.pose.pose.orientation.y, home.pose.pose.orientation.z, home.pose.pose.orientation.w]
    
    # caculate the rotation from robot to home point
    diff_quat = quaternion_multiply(home_quat, robot_quat)
    
    # Because the position is reference from robot, the real position need more transform
    diff_matrix = quaternion_matrix(diff_quat)
    offset = diff_matrix.dot(robot_dist)
    
    #Position
    msg = PoseWithCovarianceStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "map"  
    msg.pose.pose.position.x = home.pose.pose.position.x - offset[0]
    msg.pose.pose.position.y = home.pose.pose.position.y - offset[1] 
    msg.pose.pose.orientation.x = diff_quat[0]
    msg.pose.pose.orientation.y = diff_quat[1]
    msg.pose.pose.orientation.z = diff_quat[2]
    msg.pose.pose.orientation.w = diff_quat[3]
    #rospy.loginfo(msg)

    time.sleep(0.5)    
    pub.publish(msg)
  
if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
