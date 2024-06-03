#!/usr/bin/env python3
import rospy
import numpy as np
import math
import time
from sensor_msgs.msg import Imu
from std_msgs.msg import String

class Imu_angle():
    def __init__(self):
        self.turn_pub = rospy.Publisher('turn_pwm', String, queue_size=10)
        self.old_t = time.time()
        self.gyro_angle = 0.0
        self.kp_turn = 12
        self.kd_turn = 0.1
        
    def callback(self, data):
        t = time.time()
        gyro_z = data.angular_velocity.z
        acc_x = data.linear_acceleration.x
        acc_y = data.linear_acceleration.y
        angle = math.atan2(acc_x, acc_y)
        dt = t-self.old_t
        rot = gyro_z*dt
        
        self.gyro_angle += rot
        #print(self.gyro_angle)
        #e = 0.075/(0.075+dt)
        e = 0.15
        optimize_angle = (1-e) * angle + e * self.gyro_angle
        #print(optimize_angle)
        pwm = self.kp_turn * optimize_angle + self.kd_turn * gyro_z
        
        #print(round(angle*180/math.pi,2), round(optimize_angle*180/math.pi,2), "\n")
        #value = Int16()
        #value.data = 10
        self.turn_pub.publish(str(pwm))
        self.old_t = t

    
    def listener(self):
        rospy.Subscriber('imu/data', Imu, self.callback)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    listener = Imu_angle()
    listener.listener()




