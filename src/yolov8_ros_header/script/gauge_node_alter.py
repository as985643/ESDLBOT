#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
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
model = YOLO('../weights/Analog_meter_v15.pt')

MIN_SCORE = 0.4 # Minimum object detection score

# Prepare ROS
pose_pub = rospy.Publisher('object_detections', PoseArray, queue_size=10)
image_pub = rospy.Publisher('yolov8_inference_image', ROSImage, queue_size=10)
rospy.init_node('yolov8', anonymous=True)

# Prepare CvBridge
bridge = CvBridge()

# Define the frame_id as a global variable
frame_id = 0

# Define a global variable to check if the program should exit
should_exit = False

def calculate_transformation_matrix(src_points, dst_points):
    # Calculate the transformation matrix using OpenCV
    transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return transformation_matrix

def apply_transformation_matrix(image, transformation_matrix):
    # Apply the transformation matrix to the image using OpenCV
    corrected_image = cv2.warpPerspective(image, transformation_matrix, (image.shape[1], image.shape[0]))
    return corrected_image

def image_callback(msg):
    global frame_id  # Use the global keyword to refer to the global variable
    global should_exit  # Use the global keyword to refer to the global variable

    # If should_exit is True, return immediately without processing the image
    if should_exit:
        os._exit(0)
        # return

    # Convert the ROS image to OpenCV format using a cv_bridge helper function
    image_b4_color = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')


    ####################################### test #######################################
    # Define the source points (the actual perspective of the camera)
    # src_points = np.float32([[100, 100], [200, 100], [200, 200], [100, 200]])

    # # Define the destination points (the desired perspective)
    # dst_points = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])

    # # Calculate the transformation matrix
    # transformation_matrix = calculate_transformation_matrix(src_points, dst_points)

    # # Apply the transformation matrix to the image
    # image_b4_color = apply_transformation_matrix(image_b4_color, transformation_matrix)

    ####################################### test #######################################

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

    object_coordinates = []
    labels_found = []
    confidence_level_list = []
    # for prediction in predictions:
    #     box = prediction["box"]
    #     x1 = box['x1'] - box['x2']/2
    #     y1 = box['y1'] - box['y2']/2
    #     x2 = x1 + box['x2']
    #     y2 = y1 + box['y2']
    #     object_coordinates.append([x1, y1, x2, y2])
        
    #     label = prediction['name']
    #     labels_found.append(label)

    for prediction in predictions:
        label = prediction['name']
        
        # Only process the bounding boxes that are not 'The needle base' or 'The needle tip'
        if label not in ['The needle base', 'The needle tip']:
            box = prediction["box"]
            x_center = box['x1']
            y_center = box['y1']
            width = box['x2']
            height = box['y2']

            # Calculate the corners of the bounding box
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # Define the source points (the corners of the bounding box)
            src_points = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

            # Define the destination points (the desired perspective)
            # In this example, we want to transform the bounding box to a square
            dst_points = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])

            # Calculate the transformation matrix
            transformation_matrix = calculate_transformation_matrix(src_points, dst_points)

            # Apply the transformation matrix to the image
            image_b4_color = apply_transformation_matrix(image_b4_color, transformation_matrix)

    # If both Center and Needle Tip found, then calculate angle
    if "The needle base" in labels_found and "The needle tip" in labels_found:
        
        # Grabs "Center" label coordinates
        center_indexes = [index for index, x in enumerate(labels_found) if x == "The needle base"]
        center_coordinates = object_coordinates[center_indexes[0]]
        
        # Finds center x and y coordinates for "Center" label bbox
        center_x_center = int(center_coordinates[0] 
                            + (center_coordinates[2] - center_coordinates[0])/2
                            )
        center_y_center = int(center_coordinates[1] 
                            + (center_coordinates[3] - center_coordinates[1])/2
                            )
        
        # Grabs "Needle_Tip" label coordinates
        needle_tip_indexes = [index for index, x in enumerate(labels_found) if x == "The needle tip"]
        needle_tip_coordinates = object_coordinates[needle_tip_indexes[0]]
        
        # Finds center x and y coordinates for "Needle_Tip" label bbox
        center_x_needle_tip = int(needle_tip_coordinates[0] 
                                + (needle_tip_coordinates[2] - needle_tip_coordinates[0])/2
                                )
        center_y_needle_tip = int(needle_tip_coordinates[1] 
                                + (needle_tip_coordinates[3] - needle_tip_coordinates[1])/2
                                )
        
        # Finds angle - look at triginometry and arctangent
        dy = center_y_needle_tip - center_y_center
        dx = center_x_needle_tip - center_x_center
        theta = math.atan2(dy, dx)
        theta = math.degrees(theta)
        theta = round(theta)
        
        # Changes negative theta to appropriate value
        if theta < 0:
            theta *= -1
            theta = (180 - theta) + 180
        
        # Sets new starting point
        theta = theta - 90
        
        # Changes negative thetat to appropriate value
        if theta < 0:
            theta *= -1
            theta = theta + 270

        needle_tip_found = False
        pressure_gauge_4000_found = False
        pressure_gauge_200_found = False
        for label_index, label in enumerate(labels_found):
            if "The needle base" in label:
                labels_found[label_index] = label + " " +  str(theta) + " deg"
            
            if "The needle tip" in label:
                needle_tip_found = True

            if "Pressure gauge 4000" in label:
                pressure_gauge_4000_found = True

            if "Pressure gauge 200" in label:
                pressure_gauge_200_found = True

            if needle_tip_found and pressure_gauge_4000_found:
                psi = int(15.21*theta-638.21)
                labels_found[label_index] = label + " " +  str(psi) + "psi"

            if needle_tip_found and pressure_gauge_200_found:
                kPa = int(0.053*theta-2.5678)
                labels_found[label_index] = label + " " +  str(kPa) + "kPa"
    
    padding = 400  # Adjust this value to change the size of the rectangle
    # Writes text and boxes on each frame - used for boxes, degrees, and psi
    for object_coordinate_index, object_coordinate in enumerate(object_coordinates):

        start_point = ( int(object_coordinate[0]) + padding, int(object_coordinate[1]) + padding )
        color_2 = (255, 255, 255) # White

        # For text
        start_point_text = (start_point[0], max(start_point[1]-5,0) )
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        thickness = 1
        
        cv2.putText(image_b4_color, 
                    labels_found[object_coordinate_index], 
                    start_point_text, font, fontScale, color_2, thickness)

    # Convert the OpenCV image back to ROS format and publish it
    image_msg = bridge.cv2_to_imgmsg(image_b4_color, encoding='bgr8')
    image_pub.publish(image_msg)

    for prediction in predictions:
        pose = Pose()
        pose.position.x = (x1 + x2) / 2.0
        pose.position.y = (y1 + y2) / 2.0
        pose_array_msg.poses.append(pose)

    # Publish the PoseArray message
    pose_pub.publish(pose_array_msg)

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
