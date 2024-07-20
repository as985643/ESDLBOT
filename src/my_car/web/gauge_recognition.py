def gauge_node(image_path):
    import cv2
    from ultralytics import YOLO
    import ultralytics
    import numpy as np
    import math

    # Load the YOLOv8 model
    model = YOLO('static/weights/Analog_meter_v15.pt')

    # Convert the image to OpenCV format
    image_b4_color = cv2.imread(image_path)

    # Run YOLOv8 inference on the image
    results = model(image_b4_color)

    if results[0] is not None:
        predictions = results[0].tojson()
    else:
        print("results[0] is None")

    predictions = results[0].tojson()
    predictions = json.loads(predictions)

    object_coordinates = []
    labels_found = []
    for prediction in predictions:
        box = prediction["box"]
        x1 = box['x1'] - box['x2']/2
        y1 = box['y1'] - box['y2']/2
        x2 = x1 + box['x2']
        y2 = y1 + box['y2']
        object_coordinates.append([x1, y1, x2, y2])
        
        label = prediction['name']
        labels_found.append(label)

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

    return labels_found
