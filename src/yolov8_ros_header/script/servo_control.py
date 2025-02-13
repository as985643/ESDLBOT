#!/usr/bin/env python

import rospy
from std_msgs.msg import UInt16

def main():
    rospy.init_node('servo_control_node', anonymous=True)
    pan_pub = rospy.Publisher('pan_angle', UInt16, queue_size=10)
    tilt_pub = rospy.Publisher('tilt_angle', UInt16, queue_size=10)

    rate = rospy.Rate(1)  # 1 Hz

    while not rospy.is_shutdown():
        pan_angle = UInt16()
        tilt_angle = UInt16()

        # Example: Set angles to 90 degrees
        pan_angle.data = 90
        tilt_angle.data = 85

        rospy.loginfo(f"Publishing pan angle: {pan_angle.data}")
        rospy.loginfo(f"Publishing tilt angle: {tilt_angle.data}")

        pan_pub.publish(pan_angle)
        tilt_pub.publish(tilt_angle)

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
