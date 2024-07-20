#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def callback(data):
    # 使用CvBridge將ROS影像消息轉換為OpenCV影像
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

    # 顯示影像
    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

def listener():
    rospy.init_node('image_listener')
    rospy.Subscriber("/flir_lepton_image_proc/thermal_rois_body", Image, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
