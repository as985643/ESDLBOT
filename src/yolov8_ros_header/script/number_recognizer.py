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

# Load the YOLOv8 model
model = YOLO('/home/esdl/feng_ws/src/yolov8_ros_header/weights/digital_meter_v2.pt')

MIN_SCORE = 0.4 # Minimum object detection score

# Prepare ROS
pose_pub = rospy.Publisher('object_detections', PoseArray, queue_size=10)
image_pub = rospy.Publisher('yolov8_inference_image', ROSImage, queue_size=10)
rospy.init_node('number_recognizer', anonymous=True)

# Prepare CvBridge
bridge = CvBridge()

# Define the frame_id as a global variable
frame_id = 0

# Define a global variable to check if the program should exit
should_exit = False

def image_callback(msg):
    global frame_id  # Use the global keyword to refer to the global variable
    global should_exit  # Use the global keyword to refer to the global variable

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

    detected_classes = []

    # Iterate over each bounding box
    for prediction in predictions:
        # Get class name
        class_name = prediction['name']

        # If this bounding box's class is a number, '.', or '-'
        if class_name in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-']:
            # Check if 'x1' and 'y1' exist in the prediction
            if 'x1' in prediction['box'] and 'y1' in prediction['box']:
                # Add class name to detected classes
                detected_classes.append((class_name, prediction['box']['x1'], prediction['box']['y1']))

            # Draw bounding box
            x1, y1, x2, y2 = int(prediction['box']['x1']), int(prediction['box']['y1']), int(prediction['box']['x2']), int(prediction['box']['y2'])
            cv2.rectangle(image_b4_color, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Sort detected classes from left to right and top to bottom
    detected_classes.sort(key=lambda x: (x[1], x[2]))

    # Display the detected classes at the top-left corner
    cv2.putText(image_b4_color, 'Detected: %s' % ' '.join([cls[0] for cls in detected_classes]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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