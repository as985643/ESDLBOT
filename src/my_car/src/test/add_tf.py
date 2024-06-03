#!/usr/bin/env python3
import rospy
import tf

if __name__ == '__main__':
    rospy.init_node('laser_frame2usb_cam')
    cam_z = rospy.get_param('cam_z', '0.07')
    cam_x = rospy.get_param('cam_x', '0.00')
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)
    print(cam_z)
    while not rospy.is_shutdown():
        br.sendTransform((float(cam_x), 0.0, float(cam_z)),
                         (-0.5, 0.5, -0.5, 0.5),
                         rospy.Time.now(),
                         "usb_cam",
                         "base_link")
        rate.sleep()
