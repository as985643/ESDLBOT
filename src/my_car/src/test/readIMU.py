#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3


class dmp:
    def __init__(self):
        self.q = 0
        self.a = 0
        self.old_t = rospy.get_time()
        self.v = 0.0
        self.optipmize_ax = 0.0
        rospy.Subscriber('imu/data', Imu, self.callback)
            
    def dmpGetGravity(self):
        v = Vector3()
        v.x = 2 * (self.q.x*self.q.z - self.q.w*self.q.y);
        v.y = 2 * (self.q.w*self.q.x + self.q.y*self.q.z);
        v.z = self.q.w*self.q.w - self.q.x*self.q.x - self.q.y*self.q.y + self.q.z*self.q.z;
        return v;
        
    def dmpGetLinearAccel(self, gravity):
        # get rid of the gravity component (+1g = +8192 in standard DMP FIFO packet, sensitivity is 2g)
        v = Vector3()
        v.x = self.a.x - gravity.x*9.81;
        v.y = self.a.y - gravity.y*9.81;
        v.z = self.a.z - gravity.z*9.81;
        return v;

    def callback(self, data):
        self.q = data.orientation
        self.a = data.linear_acceleration
        aareal = self.realAccel()
        rospy.loginfo("%.2f", aareal.x)
        #self.filter(aareal.x)
        #rospy.loginfo("%.2f", self.optipmize_ax)
        #self.displyVelocity(aareal.x)
        
    def filter(self, new_data):
    	self.optipmize_ax = 0.75*self.optipmize_ax + 0.25*new_data

    
    def realAccel(self):
        gravity = self.dmpGetGravity()
        aareal = self.dmpGetLinearAccel(gravity)
        return aareal
        
    def displyVelocity(self, aareal):
    	t = rospy.get_time()
    	dt = t - self.old_t
    	self.v += (aareal+0.04) * dt
    	rospy.loginfo("The current velocity of x is %.2f", self.v)
    	self.old_t = t
    
if __name__ == '__main__':    
    rospy.init_node('listener', anonymous=True)
    
    dmpReal = dmp()

    rospy.spin()
