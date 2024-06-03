#!/usr/bin/env python3
import roslib
import rospy
import tf2_ros
import PyKDL
from math import pi
import sys

args = sys.argv
if len(args) > 1:
    for i in range(1, len(args)):
        param = args[i].split(":=")
        rospy.set_param(param[0], param[1])
parent_frame = rospy.get_param('parent_frame', 'imu_link')
child_frame = rospy.get_param('child_frame', 'base_link')

def listen_tf_frame(parent_frame, child_frame):
    tfbuffer = tf2_ros.Buffer()

    listener = tf2_ros.TransformListener(tfbuffer)
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            trans = tfbuffer.lookup_transform(parent_frame, child_frame, rospy.Time())
            print(trans, "\n===")
            return trans
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            #print("Fail", e)
            continue
        rate.sleep()

if __name__ == '__main__':
    #print(parent_frame)
    rospy.init_node('tf_echo')

    trans = listen_tf_frame(parent_frame, child_frame)  
    trans = trans.transform
    rot = [eval("trans.rotation."+q) for q in "xyzw"]
    print("rot", rot)
    
    #rot = PyKDL.Rotation.Quaternion(*[ eval('trans.rotation.'+c) for c in 'xyzw'] )
    #print(' '.join( [ str(eval('trans.rotation.'+c)) for c in 'xyzw'] ))
    #ypr = [ i  / pi * 180 for i in rot.GetEulerZYX() ]



