#!/usr/bin/env python3
import rospy
import numpy as np
import math
import tf2_ros
from tf.transformations import quaternion_multiply, quaternion_from_euler, euler_from_quaternion, quaternion_matrix
import PyKDL
from geometry_msgs.msg import Twist
from math import pi, atan, atan2
import time
from tf import TransformBroadcaster
import threading

def listen_tf_frame(parent_frame, child_frame):
    tfbuffer = tf2_ros.Buffer()

    listener = tf2_ros.TransformListener(tfbuffer)
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            trans = tfbuffer.lookup_transform(parent_frame, child_frame, rospy.Time())
            #print(trans, "\n===")
            return trans
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            #print("Fail", e)
            continue
        rate.sleep()
        
def listen_base_footprint_to_tag(tag_name):
    '''
    Listening until get the specify tf frame, then caculate the relative pose
    '''
    trans = listen_tf_frame('base_footprint', tag_name)
    trans_distance = [trans.transform.translation.x, trans.transform.translation.y]
    angle = atan2(trans_distance[1], trans_distance[0])
    
    trans = listen_tf_frame('odom', 'base_footprint')
    #print("trans", trans)
    base_quat = [trans.transform.rotation.x, 
                 trans.transform.rotation.y,
                 trans.transform.rotation.z,
                 trans.transform.rotation.w]
    x, y, z = euler_from_quaternion(base_quat)
    #print("z", z)
    #print("angle", angle)
    return z + angle
    
def get_base_yaw():
    '''
    Listening until get the specify tf frame, then caculate the relative pose
    '''
    trans = listen_tf_frame('odom', 'base_footprint')
    base_quat = [trans.transform.rotation.x, 
                 trans.transform.rotation.y,
                 trans.transform.rotation.z,
                 trans.transform.rotation.w]
    x, y, z = euler_from_quaternion(base_quat)
    #print(z)
    return z

def listen_deep():
    '''
    Listening until get the specify tf frame, then caculate the relative pose
    '''
    trans = listen_tf_frame('base_footprint', 'target_pos')
    deep = trans.transform.translation.x
    #print('deep', deep)

    return deep


def add_target_frame():
    std = TransformBroadcaster()
    while not rospy.is_shutdown():
    	# add target frame
        std.sendTransform(virtual_pos,
                          real_rot,
                          rospy.Time.now(),
                          "target_pos",
                          "odom")    
        rate.sleep()

def myhook():
  print("shutdown time!") 
  
if __name__ == '__main__':
    rospy.init_node('parking')
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    vel = Twist()
    rate = rospy.Rate(10.0)
    
    rospy.sleep(1)
    vel.angular.z = 0.2
    pub.publish(vel)
    rate.sleep()
    print("hi")
    listen_tf_frame('usb_cam', 'tag_0')
    vel.angular.z = 0.0
    pub.publish(vel)
    print("ya")    
    
    # find the pose of the virtual code
    real_code = listen_tf_frame("odom", "tag_0")
    real_rot = [eval("real_code.transform.rotation."+q) for q in "xyzw"]
    real_pos = [eval("real_code.transform.translation."+t) for t in "xyz"]
    real_matrix = quaternion_matrix(real_rot)
    virtual_offset = real_matrix.dot([0.0, 0.0, 0.3, 1.0])
    virtual_pos = real_pos + virtual_offset[:3]
    #print(virtual_pos)
    
    # add vitrtual target frame
    t = threading.Thread(target=add_target_frame)
    t.start()
    
    # get the target angle
    angle = listen_base_footprint_to_tag("target_pos")
    #print("need", angle)
    
    n = 0
    m = 0
    while not rospy.is_shutdown(): 
        base_angle = get_base_yaw()
        diff_angle = angle - base_angle
        #print("diff_angle", diff_angle)
        vel.angular.z = diff_angle * 1.0
        pub.publish(vel)
        if abs(diff_angle) < 0.01:
            n += 1
            if n > 5:
                vel.angular.z = 0.0
                pub.publish(vel)
                #deep = listen_deep("tag_0")
                #print("target_deep", deep)
                print('1st rotation diff', diff_angle)
                print("rotation done!")
                break
        rate.sleep()
    
    while not rospy.is_shutdown():  
        deep = listen_deep()
        if deep < 0.2:
        	vel.linear.x = deep*0.5
        else:
        	vel.linear.x = 0.1
        pub.publish(vel)
        #print("speed", speed)
        if deep < 0.0015:
            vel.linear.x = 0.0
            pub.publish(vel)
            print('position diff', deep)
            print("Done!")
            break
        rate.sleep()
    while not rospy.is_shutdown(): 
        trans = listen_tf_frame("base_footprint", "target_pos")
        base_quat = [trans.transform.rotation.x, 
                     trans.transform.rotation.y,
                     trans.transform.rotation.z,
                     trans.transform.rotation.w]
        rot = quaternion_multiply(base_quat, [-0.5, 0.5, 0.5, 0.5 ])
        x, y, z = euler_from_quaternion(rot)
        #print("z", z)
        vel.angular.z = z * 1.0
        pub.publish(vel)
        if abs(z) < 0.01:
            n += 1
            if n > 5:
                vel.angular.z = 0.0
                pub.publish(vel)
                #deep = listen_deep("tag_0")
                #print("target_deep", deep)
                print('2st rotation diff', z)
                print("rotation done!")
                break
        rate.sleep()
    rospy.single_shutdown("Done!")
        
    t.join()




