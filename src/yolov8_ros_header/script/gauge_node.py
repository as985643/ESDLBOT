#!/usr/bin/env python3
import cv2
import apriltag
from ultralytics import YOLO
import ultralytics
from PIL import Image, ImageFont, ImageDraw
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
import yaml
import tf2_ros

# Load the YOLOv8 model
model = YOLO('/home/esdl/feng_ws/src/yolov8_ros_header/weights/Analog_meter_v18.pt')

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

# Prepare tf2_ros
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

# Load the camera matrix and distortion coefficients from a previous calibration
with open('/home/esdl/.ros/camera_info/head_camera.yaml') as f:
    loadeddict = yaml.load(f, Loader=yaml.FullLoader)

mtx = np.array(loadeddict['camera_matrix']['data']).reshape(loadeddict['camera_matrix']['rows'],loadeddict['camera_matrix']['cols'])
dist = np.array(loadeddict['distortion_coefficients']['data']).reshape(-1,5)

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def check_and_correct_distortion(corners):
    # Calculate the distances between the corners
    side1 = calculate_distance(corners[0], corners[1])
    side2 = calculate_distance(corners[1], corners[2])
    side3 = calculate_distance(corners[2], corners[3])
    side4 = calculate_distance(corners[3], corners[0])
    diagonal1 = calculate_distance(corners[0], corners[2])
    diagonal2 = calculate_distance(corners[1], corners[3])

    # Check if the sides and diagonals are equal
    if not math.isclose(side1, side2) or not math.isclose(side3, side4) or not math.isclose(diagonal1, diagonal2):
        # The AprilTag is distorted, correct it
        corrected_corners = correct_perspective(corners)
        return corrected_corners
    else:
        # The AprilTag is not distorted
        return corners

def correct_perspective(image, corners):
    # Convert the corners to a list of points
    points = np.array([tuple(map(int, point)) for point in corners], dtype=np.float32)
    
    # 獲取圖像的尺寸
    (h, w) = image.shape[:2]
    
    # 定義目標座標點，這裡我們假設你希望AprilTag填滿整個影像
    # dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # 定義 AprilTag 在實際世界中的四個角點座標
    # dst_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)

    # 定義目標座標點，這裡我們將它設定為AprilTag在影像中的實際位置
    dst_points = points
    
    # 獲取透視變換矩陣
    M = cv2.getPerspectiveTransform(points, dst_points)
    
    # 執行透視變換
    corrected = cv2.warpPerspective(image, M, (w, h))
    
    return corrected


def image_callback(msg):
    global frame_id  # Use the global keyword to refer to the global variable
    global should_exit  # Use the global keyword to refer to the global variable

    # If should_exit is True, return immediately without processing the image
    if should_exit:
        os._exit(0)
        # return

    # Convert the ROS image to OpenCV format using a cv_bridge helper function
    image_b4_color = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # Undistort the image
    image_b4_color = cv2.undistort(image_b4_color, mtx, dist, None, mtx)

    # Save the image before correction
    cv2.imwrite('../img/before_correction.png', image_b4_color)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags in the image
    detector = apriltag.Detector()
    result = detector.detect(gray)

    # If an AprilTag with id 0 is detected, correct the angle
    for tag in result:
        if tag.tag_id == 1:
            print("Detected AprilTag with id 1")
            # print("tag.corners:", tag.corners)

            # Draw a rectangle around the detected AprilTag
            cv2.polylines(image_b4_color, [np.int32(tag.corners)], True, (0, 255, 0), thickness=2)

            # Listen to the transform from the camera frame to the AprilTag frame
            try:
                transform = tf_buffer.lookup_transform('usb_cam', 'tag_1', rospy.Time())
                print("AprilTag position: ", transform.transform.translation)
                print("AprilTag orientation: ", transform.transform.rotation)
                
                # Define the destination points for the perspective transform
                dst_points = np.float32([[transform.transform.translation.x, transform.transform.translation.y], 
                                         [transform.transform.translation.x + 1, transform.transform.translation.y], 
                                         [transform.transform.translation.x + 1, transform.transform.translation.y + 1], 
                                         [transform.transform.translation.x, transform.transform.translation.y + 1]])
                
                image_b4_color = correct_perspective(image_b4_color, tag.corners, dst_points)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                pass
            
            # image_b4_color = correct_perspective(image_b4_color, tag.corners)

    # Save the image after correction
    cv2.imwrite('../img/after_correction.png', image_b4_color)

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
    original_boxes =[]
    confidence_level_list = []
    for prediction in predictions:
        box = prediction["box"]
        x1 = box['x1'] - box['x2']/2
        y1 = box['y1'] - box['y2']/2
        x2 = x1 + box['x2']
        y2 = y1 + box['y2']
        object_coordinates.append([x1, y1, x2, y2])
        
        label = prediction['name']
        labels_found.append(label)

        x1_box = box['x1']
        y1_box = box['y1']
        x2_box = box['x2']
        y2_box = box['y2']
        original_boxes.append([x1_box, y1_box, x2_box, y2_box])

        # If the label is "Humidity meter", crop the image
        if label == "Humidity meter":
            # Check the coordinates of the bounding box
            if x1_box < x2_box and y1_box < y2_box:
                cropped_image = image_b4_color[int(y1_box):int(y2_box), int(x1_box):int(x2_box)]
                cv2.imshow("Cropped Image", cropped_image)
                cv2.waitKey(1)

                # Run YOLOv8 inference on the cropped image
                results = model(cropped_image)
            else:
                print("Invalid bounding box coordinates: x1_box={}, y1_box={}, x2_box={}, y2_box={}".format(x1_box, y1_box, x2_box, y2_box))

    # 繪製邊界框
    for label_index, label in enumerate(labels_found):
        # 獲取原始的邊界框座標
        x1_box, y1_box, x2_box, y2_box = original_boxes[label_index]
        
        # 將YOLO的輸出轉換為實際的像素值
        x1_pixel = int(x1_box)
        y1_pixel = int(y1_box)
        x2_pixel = int(x2_box)
        y2_pixel = int(y2_box)
        
        # 繪製邊界框
        cv2.rectangle(image_b4_color, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), (0, 255, 0), 2)
        
        # 繪製標籤
        cv2.putText(image_b4_color, label, (x1_pixel, y1_pixel - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)

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
        Humidity_meter_found = False
        Temperature_meter_50_found = False
        for label_index, label in enumerate(labels_found):
            if "The needle tip" in label:
                needle_tip_found = True

            if "Pressure gauge 4000" in label:
                pressure_gauge_4000_found = True

            if "Pressure gauge 200" in label:
                pressure_gauge_200_found = True
            
            if "Humidity meter" in label:
                Humidity_meter_found = True

            if "Temperature meter 50" in label:
                Temperature_meter_50_found = True

            if needle_tip_found and pressure_gauge_4000_found:
                psi = int(15.21*theta-638.21)
                labels_found[label_index] = label + " " +  str(psi) + "psi"

            if needle_tip_found and pressure_gauge_200_found:
                kPa = int(0.053*theta-2.5678)
                labels_found[label_index] = label + " " +  str(kPa) + "kPa"
            
            if needle_tip_found and Humidity_meter_found:
                percentage = int(0.2975*theta+3.038)
                labels_found[label_index] = label + " " +  str(percentage) + "percentage"

            if needle_tip_found and Temperature_meter_50_found:
                degrees = int(0.3185*theta-47.29)
                labels_found[label_index] = label + " " +  str(degrees) + "degrees"

            # Writes text and boxes on each frame - used for boxes, degrees, and psi
            for object_coordinate_index, object_coordinate in enumerate(object_coordinates):
                color_2 = (255, 255, 255) # White

                # If both Needle Tip and Pressure Gauge 4000 or Pressure Gauge 200 are found, then draw psi or kPa at the top left corner
                if needle_tip_found and (pressure_gauge_4000_found or pressure_gauge_200_found):
                    # For text
                    start_point_text = (10, 30)  # Top left corner of the screen
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    thickness = 2
                    text = 'PSI: %d' % psi if pressure_gauge_4000_found else 'kPa: %d' % kPa
                
                    cv2.putText(image_b4_color, 
                                text, 
                                start_point_text, font, fontScale, (0, 255, 0), thickness)

                # If both Needle Tip and Humidity meter are found, then draw percentage at the top left corner
                if needle_tip_found and Humidity_meter_found:
                    # For text
                    start_point_text = (10, 60)  # Top left corner of the screen
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    thickness = 2
                    text = 'Humidity: %d%%' % percentage
                
                    cv2.putText(image_b4_color, 
                                text, 
                                start_point_text, font, fontScale, (0, 255, 0), thickness)

                # If both Needle Tip and Temperature meter 50 are found, then draw degrees at the top left corner
                # if needle_tip_found and Temperature_meter_50_found:
                #     # For text
                #     start_point_text = (10, 90)  # Top left corner of the screen
                #     font = cv2.FONT_HERSHEY_SIMPLEX
                #     fontScale = 1
                #     thickness = 2
                #     text = 'Temperature: %d degrees' % degrees
                
                #     cv2.putText(image_b4_color, 
                #                 text, 
                #                 start_point_text, font, fontScale, (0, 255, 0), thickness)

                # If both Needle Tip and Temperature meter 50 are found, then draw degrees at the top left corner
                if needle_tip_found and Temperature_meter_50_found:
                    # For text
                    start_point_text = (10, 0)  # Top left corner of the screen
                    fontpath = "/home/esdl/.pyenv/versions/3.9.0/envs/ocr/lib/python3.9/site-packages/cv2/qt/fonts/DejaVuSans.ttf"
                    font = ImageFont.truetype(fontpath, 32)
                    image_pil = Image.fromarray(image_b4_color)
                    draw = ImageDraw.Draw(image_pil)
                    text = 'Temperature: %d \u00B0C' % degrees

                    # Draw the text on the image
                    draw.text(start_point_text, text, font=font, fill=(0, 255, 0))

                    # Convert the PIL image back to a numpy array
                    image_b4_color = np.array(image_pil)
            


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
