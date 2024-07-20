#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ImageSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("usb_cam/image_raw/compressed", CompressedImage, self.image_callback)
        self.save_image = False
        self.cv_image = None
        self.image_count = 0

    def image_callback(self, data):
        try:
            np_arr = np.frombuffer(data.data, np.uint8)
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            print(e)

        if self.save_image:
            cv2.imwrite('../img/image{}.jpg'.format(self.image_count), self.cv_image)
            self.image_count += 1
            self.save_image = False

    def check_for_input(self):
        if self.cv_image is not None:
            cv2.imshow('Image Window', self.cv_image)

        key = cv2.waitKey(3)
        if key == ord('s'):
            self.save_image = True
        elif key == ord('q'):
            cv2.destroyAllWindows()
            rospy.signal_shutdown('User requested shutdown')

def main():
    rospy.init_node('image_saver', anonymous=True)
    image_saver = ImageSaver()

    rate = rospy.Rate(10) # 10 Hz
    while not rospy.is_shutdown():
        image_saver.check_for_input()
        rate.sleep()

if __name__ == '__main__':
    main()
