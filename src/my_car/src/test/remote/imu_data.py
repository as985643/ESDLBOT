#!/usr/bin/env python3
import rospy
import numpy as np
import math
import time
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3

class Imu_data():
    def __init__(self):
        self.simple_imu_pub = rospy.Publisher('simple_imu', Vector3, queue_size=10)
        self.old_t = time.time()
        self.gyro_angle = 0.0
        self.kp_turn = 12
        self.kd_turn = 0.1
        
    def callback(self, data):
        t = time.time()
        gyro_z = data.angular_velocity.z
        acc_x = data.linear_acceleration.x
        #acc_y = data.linear_acceleration.y
        #angle = math.atan2(acc_x, acc_y)
        dt = t-self.old_t
        #rot = gyro_z*dt
        
        #self.gyro_angle += rot
        #print(self.gyro_angle)
        #e = 0.075/(0.075+dt)
        #optimize_angle = (1-e) * angle + e * self.gyro_angle
        #print(optimize_angle)
        msg = Vector3()
        msg.x = acc_x
        msg.y = gyro_z
        msg.z = dt
        
        #print(round(angle*180/math.pi,2), round(optimize_angle*180/math.pi,2), "\n")
        #value = Int16()
        #value.data = 10
        self.simple_imu_pub.publish(msg)
        self.old_t = t

    
    def listener(self):
        rospy.Subscriber('imu/data', Imu, self.callback)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    listener = Imu_data()
    listener.listener()




