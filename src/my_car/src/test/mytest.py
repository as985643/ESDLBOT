#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import tf2_ros
import geometry_msgs.msg

try:
    data = rospy.get_param('~data')
except:
    data = 100
    

class test:
    def __init__(self):
        self.data = data
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        
    def set_frame(self):

        static_transformStamped = geometry_msgs.msg.TransformStamped()
   
        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "map"
        static_transformStamped.child_frame_id = "test"
        static_transformStamped.transform.translation.x = 0.0
        static_transformStamped.transform.translation.y = 0.0
        static_transformStamped.transform.translation.z = 0.5
   
        static_transformStamped.transform.rotation.x = 0.0
        static_transformStamped.transform.rotation.y = 0.0
        static_transformStamped.transform.rotation.z = 0.0
        static_transformStamped.transform.rotation.w = 1.0
        
        #self.broadcaster.sendTransform(static_transformStamped)
        self.broadcaster.sendTransform([0, 0, 1],
                                       [0, 0, 0, 1],
                                       rospy.Time.now(),
                                       "target_dummy",
                                       "map")
                                                 
    def main(self):
            rospy.loginfo("I read the data : %d", self.data)
            rospy.sleep(1)
    
if __name__ == '__main__':    
    rospy.init_node('listener', anonymous=True)
    
    mytest = test()
    mytest.set_frame()
    #while not rospy.is_shutdown():
    	#mytest.main()
    rospy.spin()
