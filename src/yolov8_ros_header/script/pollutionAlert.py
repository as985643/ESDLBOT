#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
import ultralytics
from PIL import Image
import io
import os
import sys
import json
import numpy as np
import math
import rospy
from std_msgs.msg import String 
from geometry_msgs.msg import Point, PoseArray, Pose,Quaternion
from std_msgs.msg import Header
from sensor_msgs.msg import Image as ROSImage, CompressedImage
from cv_bridge import CvBridge
from linebot import LineBotApi
from linebot.models import TextSendMessage
import time

# Load the YOLOv8 model
model = YOLO('/home/esdl/feng_ws/src/yolov8_ros_header/weights/pollution.pt')

MIN_SCORE = 0.4 # Minimum object detection score

# Prepare ROS
pose_pub = rospy.Publisher('object_detections', PoseArray, queue_size=10)
image_pub = rospy.Publisher('yolov8_inference_image', ROSImage, queue_size=10)
rospy.init_node('pollution_detector', anonymous=True)

# Prepare CvBridge
bridge = CvBridge()

# Define the frame_id as a global variable
frame_id = 0

# Define a global variable to check if the program should exit
should_exit = False

# Prepare Line Notify
line_bot_api = LineBotApi('nZCXfUdx1HNzilxLWjHbTicAoIPJ9MI710D9JCtgmBF')

# Initialize counter and timestamp
counter = 0
last_notify_time = time.time()

# Set the notification interval (in seconds)
notify_interval = 60 * 5  # 5 minutes

# Set the notification threshold
notify_threshold = 10

def image_callback(msg):
    global frame_id  # Use the global keyword to refer to the global variable
    global should_exit  # Use the global keyword to refer to the global variable
    global counter  # Use the global keyword to refer to the global variable
    global last_notify_time  # Use the global keyword to refer to the global variable

    # If should_exit is True, return immediately without processing the image
    if should_exit:
        os._exit(0)

    # Convert the ROS image to OpenCV format using a cv_bridge helper function
    image_b4_color = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # Run YOLOv8 inference on the frame
    results = model(image_b4_color)

    # Create a PoseArray message
    pose_array_msg = PoseArray()
    pose_array_msg.header = Header()
    pose_array_msg.header.stamp = rospy.Time.now()

    # Define the frame_id
    pose_array_msg.header.frame_id = str(frame_id)

    if results[0] is not None:
        predictions = results[0].tojson()
    else:
        print("results[0] is None")

    predictions = results[0].tojson()
    predictions = json.loads(predictions)

    # Initialize counter
    count = 0
    detected_classes = []

    # Iterate over each bounding box
    for prediction in predictions:
        # Get class name
        class_name = prediction['name']
        detected_classes.append(class_name)

        # If this bounding box's class is 'Pollution'
        if class_name == 'pollution':
            # Increment counter
            counter += 1

            # Draw bounding box
            x1, y1, x2, y2 = int(prediction['box']['x1']), int(prediction['box']['y1']), int(prediction['box']['x2']), int(prediction['box']['y2'])
            cv2.rectangle(image_b4_color, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Check if it's time to send a notification
            if counter >= notify_threshold and time.time() - last_notify_time >= notify_interval:
                # Send Line Notify
                line_bot_api.push_message('nZCXfUdx1HNzilxLWjHbTicAoIPJ9MI710D9JCtgmBF', TextSendMessage(text='Detected %d pollution sources' % counter))

                # Reset counter and update timestamp
                counter = 0
                last_notify_time = time.time()

    # Convert the OpenCV image back to ROS format and publish it
    image_msg = bridge.cv2_to_imgmsg(image_b4_color, encoding='bgr8')
    image_pub.publish(image_msg)

    # Increment the frame_id
    frame_id += 1

    # Display the image with bounding boxes
    cv2.imshow("YOLOv8 Inference", image_b4_color)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        should_exit = True

rospy.Subscriber('usb_cam/image_raw/compressed', CompressedImage, image_callback)

while not rospy.is_shutdown() and not should_exit:
    rospy.spin()
