#!/usr/bin/env python3
import cv2
from ultralytics import YOLO, solutions
from PIL import Image
import io
import os
import sys
import json
import numpy as np
import math
import rospy
from std_msgs.msg import String 
from geometry_msgs.msg import Point, PoseArray, Pose, Quaternion
from std_msgs.msg import Header
from sensor_msgs.msg import Image as ROSImage, CompressedImage
from cv_bridge import CvBridge
import apriltag
import time
import datetime

# Load the YOLOv8 model
model = YOLO('/home/esdl/feng_ws/src/yolov8_ros_header/weights/count_mushrooms_v3.pt').to('cuda')
print(next(model.parameters()).device)  # 應該輸出 cuda:0 或其他 GPU 設備

MIN_SCORE = 0.4 # Minimum object detection score

# Prepare ROS
pose_pub = rospy.Publisher('object_detections', PoseArray, queue_size=10)
image_pub = rospy.Publisher('yolov8_inference_image', ROSImage, queue_size=10)
rospy.init_node('mushroom_counter', anonymous=True)

# Prepare CvBridge
bridge = CvBridge()

# Define the frame_id as a global variable
frame_id = 0

# Define global variables to check if the program should exit and for analytics
should_exit = False
total_counts = 0
frame_count = 0
cumulative_count = 0  # Define a global variable for cumulative count

# Initialize the detector
detector = apriltag.Detector()

# 設定AprilTag的實際尺寸
april_tag_size_cm = 4.0

# Initialize video writer
out = None
analytics = None

# Set to keep track of counted object IDs
counted_ids = set()

# Function to get the next available filename
def get_next_filename(directory, extension):
    i = 1
    while os.path.exists(f"{directory}/{i}.{extension}"):
        i += 1
    return f"{directory}/{i}.{extension}"

# Function to get the next available filename with current time
def get_next_filename_with_time(directory, extension):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{directory}/{current_time}.{extension}"

def image_callback(msg):
    global frame_id, should_exit, total_counts, frame_count, out, analytics, counted_ids, cumulative_count

    # If should_exit is True, return immediately without processing the image
    if should_exit:
        os._exit(0)

    # Convert the ROS image to OpenCV format using a cv_bridge helper function
    image_b4_color = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # Initialize video writer and analytics if not already done
    if out is None or analytics is None:
        h, w = image_b4_color.shape[:2]
        fps = 30  # Assuming a default FPS value
        video_path = get_next_filename_with_time("video", "avi")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        analytics = solutions.Analytics(
            type="line",
            writer=out,
            im0_shape=(w, h),
            view_img=True,
            x_label="Frame Count (fps)",  # 設置 x 軸標籤
            y_label="Detected Objects Count"  # 設置 y 軸標籤
        )

    # Detect the AprilTags in the image
    gray = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2GRAY)
    result = detector.detect(gray)

    mm_per_pixel = None
    if result:
        # 偵測到AprilTag
        for r in result:
            # 獲取AprilTag的邊長（以像素為單位）
            (ptA, ptB, ptC, ptD) = r.corners
            ptA = (int(ptA[0]), int(ptA[1]))
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))

            # 繪製AprilTag的邊界框
            cv2.line(image_b4_color, ptA, ptB, (0, 255, 0), 2)
            cv2.line(image_b4_color, ptB, ptC, (0, 255, 0), 2)
            cv2.line(image_b4_color, ptC, ptD, (0, 255, 0), 2)
            cv2.line(image_b4_color, ptD, ptA, (0, 255, 0), 2)

            # 計算AprilTag的邊長（以像素為單位）
            pixel_width = np.linalg.norm(np.array(ptA) - np.array(ptB))

            # 計算比例因子
            mm_per_pixel = (april_tag_size_cm * 10) / pixel_width

            # 顯示AprilTag的實際寬度
            cv2.putText(image_b4_color, f'{april_tag_size_cm * 10:.2f} mm', (ptA[0], ptA[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print(f"每像素的毫米數: {mm_per_pixel:.4f} mm")

    # 計算推論時間
    start_time = time.time()

    # Run YOLOv8 inference on the whole image
    results = model.track(image_b4_color, persist=True, verbose=True, device='cuda')

    # 計算推論時間
    inference_time_ms = (time.time() - start_time) * 1000

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
        # Get class name and object ID
        class_name = prediction['name']
        object_id = prediction.get('track_id', None)  # Use 'track_id' instead of 'tracking_id'
        detected_classes.append(class_name)

        # If this bounding box's class is 'King oyster mushroom'
        if class_name == 'King_oyster_mushroom':
            # Draw bounding box
            x1, y1, x2, y2 = int(prediction['box']['x1']), int(prediction['box']['y1']), int(prediction['box']['x2']), int(prediction['box']['y2'])
            cv2.rectangle(image_b4_color, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 計算物件的實際寬度
            if mm_per_pixel is not None:
                object_pixel_width = x2 - x1
                object_real_width_mm = object_pixel_width * mm_per_pixel
                cv2.putText(image_b4_color, f'{object_real_width_mm:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # If it hasn't been counted yet
            if object_id is not None and object_id not in counted_ids:
                # Increment counter
                count += 1
                counted_ids.add(object_id)

    # Update cumulative count
    cumulative_count += count

    rospy.loginfo('Detected %d King oyster mushrooms', count)

    # Display the cumulative count at the top-left corner
    cv2.putText(image_b4_color, 'Count: %d' % cumulative_count, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 顯示推論速度在右上角
    cv2.putText(image_b4_color, f'{inference_time_ms:.2f} ms', (image_b4_color.shape[1] - 80, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Convert the OpenCV image back to ROS format and publish it
    image_msg = bridge.cv2_to_imgmsg(image_b4_color, encoding='bgr8')
    image_pub.publish(image_msg)

    # Increment the frame_id
    frame_id += 1

    # Display the image with bounding boxes
    # cv2.imshow("YOLOv8 Inference", image_b4_color)

    window_name = "YOLOv8 Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 獲取螢幕解析度
    screen_width = 1920  # 替換為你的螢幕寬度
    screen_height = 1080  # 替換為你的螢幕高度

    # 計算視窗位置，使其顯示在螢幕中間
    window_width = image_b4_color.shape[1]
    window_height = image_b4_color.shape[0]
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    # 移動視窗到指定位置
    cv2.moveWindow(window_name, x, y)

    cv2.imshow(window_name, image_b4_color)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        should_exit = True

    # Update analytics
    frame_count += 1
    analytics.update_line(frame_count, count)

rospy.Subscriber('usb_cam/image_raw/compressed', CompressedImage, image_callback)

while not rospy.is_shutdown() and not should_exit:
    rospy.spin()

if out is not None:
    out.release()
cv2.destroyAllWindows()
