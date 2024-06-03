#!/usr/bin/env python3
import rospy
import tf
from tf2_msgs.msg import TFMessage
import math
import time

def callback(data):
    CodeInfo = data.transforms[0] 
    CodeID = CodeInfo.child_frame_id
    if CodeID == "tag_0":
        #rospy.loginfo('I heard %s', data.transforms[0].child_frame_id)
        translation = CodeInfo.transform.translation
        Code_Quaternion = (CodeInfo.transform.rotation.x,
                      CodeInfo.transform.rotation.y,
                      CodeInfo.transform.rotation.z,
                      CodeInfo.transform.rotation.w)
        #Code_Quaternion = CodeInfo.transform.rotation
        #rospy.loginfo(Code_Quaternion)
        euler = [round(x/math.pi*180, 2) for x in tf.transformations.euler_from_quaternion(Code_Quaternion)]
        print("\nRPY :\n")
        rospy.loginfo(euler)
def listener():
 
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    
    rospy.Subscriber('tf', TFMessage, callback)
    
    print("hi!")
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    print("ya")
if __name__ == '__main__':
    listener()
