#!/usr/bin/env python3
import rospy
import numpy as np
import math
import time
from sensor_msgs.msg import Imu
from std_msgs.msg import String

class Imu_angle():
    def __init__(self):
        self.pub = rospy.Publisher('pwm', String, queue_size=10)
        self.old_t = time.time()
        self.gyro_angle = 0.0
        self.old_bias = 0.0
        
    def callback(self, data):
        t = time.time()
        gyro_y = data.angular_velocity.y
        acc_x = data.linear_acceleration.x
        acc_z = data.linear_acceleration.z
        angle = math.atan2(acc_x, acc_z)

        rot = gyro_y*(t-self.old_t)
        self.gyro_angle += rot
        
        optimize_angle = 0.98 * angle + 0.02 * self.gyro_angle
        
        pwm = (optimize_angle-self.old_bias)*120
        
        #print(round(angle*180/math.pi,2), round(optimize_angle*180/math.pi,2), "\n")
        #value = Int16()
        #value.data = 10
        self.pub.publish(str(pwm))
        self.old_t = t
        self.old_bias = optimize_angle
        #rospy.sleep(1)
        
    
    def listener(self):
        rospy.Subscriber('imu/data', Imu, self.callback)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    listener = Imu_angle()
    listener.listener()




