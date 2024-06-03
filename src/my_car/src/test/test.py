#!/usr/bin/env python3
import rospy
from std_msgs.msg import String



class test:
    def __init__(self):
        self.data = data
        self.node_name = rospy.get_name()
            
    def main(self):
            rospy.loginfo("%s read the data : %d", self.node_name, self.data)
            if rospy.get_name() == '/pkg1':
            	rospy.loginfo(True)
            rospy.sleep(1)
    
if __name__ == '__main__':    
    rospy.init_node('listener', anonymous=True)
    data = rospy.get_param('~data', 100)
    mytest = test()
    while not rospy.is_shutdown():
    	mytest.main()
    #rospy.spin()
