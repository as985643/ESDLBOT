from flask import Flask, render_template, url_for,g,jsonify,request,make_response,send_from_directory,redirect,flash
import subprocess
import signal
import os
import time
from mySQL import mySQL

from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
import glob
from PIL import Image
import shutil
import json
import psycopg2
import math
import numpy as np

app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024

car_sensor = mySQL("car_sensor")
maps = mySQL("maps")

DATABASE_URL = {"database":"esdl", "user":"esdl", "password":"bj/6m06",
                "host":"192.168.1.180", "port":"5432"}		# sql 資訊

# DATABASE_URL = {"database":"esdl", "user":"esdl", "password":"bj/6m06",
#                 "host":"192.168.67.102", "port":"5432"}		# sql 資訊

class roslaunch_process():
    @classmethod
    def start_navigation(self,mapname):
        self.process_navigation = subprocess.Popen(["roslaunch","--wait", "my_car", "esdlbot_V1_navigation.launch", "map_file:="+os.getcwd()+"/static/maps/"+mapname+".yaml"])

    @classmethod
    def stop_navigation(self):
        try:
            self.process_navigation.send_signal(signal.SIGINT)	
        except:
            pass

    @classmethod
    def start_mapping(self):
        self.process_mapping = subprocess.Popen(["roslaunch", "--wait", "my_car", "backpack_2d.launch"])

    @classmethod
    def stop_mapping(self):

        try: 
            self.process_mapping.send_signal(signal.SIGINT)    
        except: 
            pass


@app.teardown_appcontext
def close_connection(exception):
    print("Close connect!")
    #sql.close_sql()

'''
@app.before_first_request
def create_table():
    subprocess.Popen(["ssh", "ubuntu@192.168.1.192"])

    subprocess.Popen(["roslaunch", "turtlebot3_navigation", "turtlebot3_bringup.launch"])
    

    with app.app_context():
	    try:
	        c = get_db().cursor()
	        c.execute("CREATE TABLE IF NOT EXISTS maps (id integer PRIMARY KEY,name text NOT NULL)")
	        c.close()
	    except Error as e:
	        print(e)
'''




@app.route('/', methods = ['POST', 'GET'])
def index():

	#subprocess.Popen(["rosrun", "tf2_web_republisher", "tf2_web_republisher"])
	
	## Get local ipv4 ##
	import socket
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.connect(("8.8.8.8", 80))
	ip = s.getsockname()[0]
	s.close()
	
	maps_list = maps.read_sql()[1]
	resp = make_response(render_template('index.html',title='ESDLBOT',maps = maps_list))
	resp.set_cookie('serverip', ip)

	return resp

@app.route('/sqlview')
def sqlview():
	
	# with get_db():
	# 	try:
	# 		c = get_db().cursor()
	# 		c.execute("SELECT * FROM maps")
	# 		data = c.fetchall()
	# 		c.close()
	# 	except Error as e:
	# 		print(e)
	'''
	list_users = [  [0, 'A1', 2, 3, 8.2,'2022-11-3 11:59:01'],
	                [1, 'A2', 6, 7, 7.3, '2022-11-3 11:59:01'],
	                [2, 'A1', 10, 11, 6.5, '2022-11-3 11:59:01'],
	                [3, 'A3', 6, 7, 4.8, '2022-11-3 11:59:01'],
	                [4, 'A2', 6, 7, 2.5, '2022-11-3 11:59:01'],
	                [5, 'A1', 10, 11, 3.3, '2022-11-3 11:59:01'],
	                [6, 'A3', 6, 7, 5.5, '2022-11-3 11:59:01'],
	                [7, 'A2', 10, 11, 8.9, '2022-11-3 11:59:01'],
	                [8, 'A1', 6, 7, 4.1, '2022-11-3 11:59:01'],
	                [9, 'A3', 10, 11, 5.6, '2022-11-3 11:59:01'],
	                [10, 'A3', 10, 11, 1.2, '2022-11-3 11:59:01'],
	                [11, 'A1', 6, 7, 8.8, '2022-11-3 11:59:01'],
	                [12, 'A2', 10, 11, 6.5, '2022-11-3 11:59:01'],
	                [13, 'A4', 14, 15, 7.2, '2022-11-3 11:59:01']]'''
	list_users = car_sensor.read_sql()[1]
	maps_list = maps.read_sql()[1]
	return render_template('SQLView.html', title='SQL View', list_users = list_users, maps=maps_list)

# 伺服器端
@app.route('/get_table_data', methods=['GET'])
def get_table_data():
    # 從資料庫中獲取表格的資料
    # list_users = car_sensor.read_sql()
    # return jsonify(list_users)
    try:
        # 從資料庫中獲取表格的資料
        list_users = car_sensor.read_sql()
        return jsonify(list_users)
    except Exception as e:
        print(f"錯誤: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/edit/<int:record_no>', methods=['GET', 'POST'])
def edit(record_no):
    # Connect to your postgres DB
    conn = psycopg2.connect(**DATABASE_URL, sslmode='require')
    cur = conn.cursor()

    if request.method == 'POST':
        new_area = request.form['area']
        cur.execute("UPDATE car_sensor SET area = %s WHERE record_no = %s", (new_area, record_no))
        conn.commit()
        return redirect(url_for('sqlview'))

    cur.execute("SELECT area FROM car_sensor WHERE record_no = %s", (record_no,))
    area = cur.fetchone()[0]
    return render_template('edit.html', area=area)

@app.route('/sql_upload', methods=['GET'])
def handle_request():
    record_no = request.args.get('record_no')
    area = request.args.get('area')
    temperature = request.args.get('temperature')
    humidity = request.args.get('humidity')
    co2 = request.args.get('co2')
    time = request.args.get('time')

    # Connect to your postgres DB
    conn = psycopg2.connect(**DATABASE_URL, sslmode='require')
    cur = conn.cursor()

    # Find the maximum record_no in the database
    cur.execute("SELECT MAX(record_no) FROM car_sensor")
    max_record_no = cur.fetchone()[0]

    # If the table is empty, start from 1, otherwise increment the maximum record_no
    if max_record_no is None:
        record_no = '1'
    else:
        record_no = str(int(max_record_no) + 1)

    # Insert the sensor value into the DB
    cur.execute("INSERT INTO car_sensor (record_no, area, temperature, humidity, co2, time) VALUES (%s, %s, %s, %s, %s, %s)", 
				(record_no, area, temperature, humidity, co2, time))
    conn.commit()

    cur.close()
    conn.close()

    return 'OK', 200

@app.route('/delete/<int:record_no>', methods=['POST'])
def delete(record_no):
    # Connect to your postgres DB
    conn = psycopg2.connect(**DATABASE_URL, sslmode='require')
    cur = conn.cursor()

    # Delete the record from the database
    cur.execute("DELETE FROM car_sensor WHERE record_no = %s", (record_no,))
    conn.commit()

    cur.close()
    conn.close()

    return redirect(url_for('sqlview'))  # Redirect to SQLView.html


@app.route('/add_student', methods=['POST'])
def add_student():
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    if request.method == 'POST':
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['email']
        cur.execute("INSERT INTO students (fname, lname, email) VALUES (%s,%s,%s)", (fname, lname, email))
        conn.commit()
        flash('Student Added successfully')						
        return redirect(url_for('Index'))


@app.route('/robotControl')
def robotControl():
	maps_list = maps.read_sql()[1]
	return render_template('robotControl.html', title='Robot Control', maps=maps_list)

# @app.route('/AIdetect')
# def AIdetect():
# 	maps_list = maps.read_sql()[1]
# 	return render_template('AIdetect.html', title='AIdetect', maps=maps_list)

# def correct_image(image, box_coordinates):
#     # Get the coordinates of the bounding box
#     x1, y1, x2, y2 = box_coordinates

#     # Calculate the center of the bounding box
#     center_x = (x1 + x2) / 2
#     center_y = (y1 + y2) / 2

#     # Calculate the width and height of the bounding box
#     width = x2 - x1
#     height = y2 - y1

#     # Calculate the angle of rotation
#     angle = np.arctan2(height, width)

#     # Create a rotation matrix
#     M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)

#     # Apply the rotation to the image
#     corrected_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

#     return corrected_image

@app.route('/AIdetect')
def upload_file():
   return render_template('uploads.html',title='AIdetect')
# @app.route('/standby', methods=['POST'])
# def standby():
#     command = "rosrun mycobot_startup_pose startup_pose.py"
#     process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
#     process.wait()
#     return "Node started!"

@app.route('/AIdetect', methods = ['GET', 'POST'])
def uploads():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        filepath = os.path.join('./static/uploads', filename)
        f.save(filepath)

        url_for_filepath = filepath.replace('./static/', '')

        model_choice = request.form.get('model_choice')

        if model_choice == 'Analog_meter':
            model_path = 'static/weights/Analog_meter_v18.pt'
        elif model_choice == 'digital_meter':
            model_path = 'static/weights/digital_meter_v3.pt'
        elif model_choice == 'count_mushrooms':
            model_path = 'static/weights/count_mushrooms_v4.pt'
        elif model_choice == 'pollution':
            model_path = 'static/weights/pollution.pt'
        else:
            return '未知的模型選擇', 400

        model = YOLO(model_path)

        result_image = model([filepath], save=True)

        predictions = result_image[0].tojson()
        predictions = json.loads(predictions)

        predictions.sort(key=lambda x: (x['box']['x1'], x['box']['y1']))

        count = 0
        detected_classes = []
        transcription_results = []

        for prediction in predictions:
            class_name = prediction['name']
            detected_classes.append(class_name)
            if class_name == 'King_oyster_mushroom':
                count += 1

        folders = glob.glob('runs/detect/predict*')
        latest_folder = max(folders, key=os.path.getmtime)
        result_image = os.path.join(latest_folder, filename)

        dst_path = os.path.join('./static/results/', filename)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        shutil.move(result_image, './static/results/')
        result_image = os.path.join('results', filename)

        model_gauge = YOLO('static/weights/Analog_meter_v18.pt')
        image_b4_color = cv2.imread(filepath)

        results_gauge = model_gauge(image_b4_color)

        if results_gauge[0] is not None:
            predictions_gauge = results_gauge[0].tojson()
        else:
            print("results_gauge[0] is None")

        predictions_gauge = results_gauge[0].tojson()
        predictions_gauge = json.loads(predictions_gauge)

        object_coordinates_gauge = []
        labels_found_gauge = []
        for prediction in predictions_gauge:
            box = prediction["box"]
            x1 = box['x1'] - box['x2']/2
            y1 = box['y1'] - box['y2']/2
            x2 = x1 + box['x2']
            y2 = y1 + box['y2']
            object_coordinates_gauge.append([x1, y1, x2, y2])

            # meter_box_coordinates = [x1, y1, x2, y2]

            label = prediction['name']
            labels_found_gauge.append(label)

        # corrected_image = correct_image(image_b4_color, meter_box_coordinates)
        # results_gauge = model_gauge(corrected_image)

        if "The needle base" in labels_found_gauge and "The needle tip" in labels_found_gauge:
            center_indexes = [index for index, x in enumerate(labels_found_gauge) if x == "The needle base"]
            center_coordinates = object_coordinates_gauge[center_indexes[0]]
            center_x_center = int(center_coordinates[0] + (center_coordinates[2] - center_coordinates[0])/2)
            center_y_center = int(center_coordinates[1] + (center_coordinates[3] - center_coordinates[1])/2)

            needle_tip_indexes = [index for index, x in enumerate(labels_found_gauge) if x == "The needle tip"]
            needle_tip_coordinates = object_coordinates_gauge[needle_tip_indexes[0]]
            center_x_needle_tip = int(needle_tip_coordinates[0] + (needle_tip_coordinates[2] - needle_tip_coordinates[0])/2)
            center_y_needle_tip = int(needle_tip_coordinates[1] + (needle_tip_coordinates[3] - needle_tip_coordinates[1])/2)

            dy = center_y_needle_tip - center_y_center
            dx = center_x_needle_tip - center_x_center
            theta = math.atan2(dy, dx)
            theta = math.degrees(theta)
            theta = round(theta)

            if theta < 0:
                theta *= -1
                theta = (180 - theta) + 180

            theta = theta - 90

            if theta < 0:
                theta *= -1
                theta = theta + 270

            needle_tip_found = False
            pressure_gauge_4000_found = False
            pressure_gauge_200_found = False
            Pressure_gauge_140_found = False
            Humidity_meter_found = False
            Temperature_meter_50_found = False
            for label_index, label in enumerate(labels_found_gauge):
                if "The needle tip" in label:
                    needle_tip_found = True

                if "Pressure gauge 4000" in label:
                    pressure_gauge_4000_found = True

                if "Pressure gauge 200" in label:
                    pressure_gauge_200_found = True
                    
                if "Pressure gauge 140" in label:
                    Pressure_gauge_140_found = True

                if "Humidity meter" in label:
                    Humidity_meter_found = True

                if "Temperature meter 50" in label:
                    Temperature_meter_50_found = True

            if needle_tip_found and pressure_gauge_4000_found:
                psi = int(15.21*theta-638.21)
                transcription_results.append(str(psi) + "psi")

            if needle_tip_found and pressure_gauge_200_found:
                kPa = int(0.053*theta-2.5678)
                transcription_results.append(str(kPa) + "kPa")
                
            if needle_tip_found and Pressure_gauge_140_found:
                psi = int(0.5942*theta-23.65)  # 這裡的公式可能需要根據實際情況進行調整
                transcription_results.append(str(psi) + "psi")
                
            if needle_tip_found and Temperature_meter_50_found:
                degrees = int(0.3185*theta-47.29)
                transcription_results.append( "Temperature: "+str(degrees) + "°C")

            if needle_tip_found and Humidity_meter_found:
                percentage = int(0.2975*theta+3.038)
                transcription_results.append( "Humidity: "+ str(percentage) + "%")

            if needle_tip_found and "Pressure gauge 160" in label:
                kPa = int(15.21*theta-638.21)  # 這裡的公式可能需要根據實際情況進行調整
                transcription_results.append(str(kPa) + "kPa")

        return render_template('results.html', result_image=result_image, mushroom_count=count,
                                original_image=url_for_filepath, detected_classes=detected_classes,
                                transcription_results=transcription_results, title='AIdetect')
    else:
        return render_template('uploads.html',title='AIdetect')

@app.route('/pathplan')
def pathplan():
	maps_list = maps.read_sql()[1]
	return render_template('pathplan.html', title='Path Plan', maps=maps_list)

@app.route('/webimage')
def webimage():
	maps_list = maps.read_sql()[1]
	return render_template('webimage.html', title='Camera', maps=maps_list)

@app.route('/index/<variable>',methods=['GET','POST'])
def themainroute(variable):
	if variable == "navigation-precheck" :
		'''with get_db():


	        	try:
	        
		            c = get_db().cursor()
		           
		            c.execute("SELECT count(*) FROM maps")
		            k=c.fetchall()[0][0]
		            c.close()
		           
		            print(k)
		            return jsonify(mapcount=k) 
	                
	            
	        	except Error as e:
	            		print(e)
		'''
		return jsonify(mapcount=1) 
        
	elif variable == "gotonavigation":

		mapname =request.get_data().decode('utf-8')

		roslaunch_process.start_navigation(mapname)
		
		return "success"



      

		    
@app.route('/navigation',methods=['GET','POST'])

def navigation():

	
	# with get_db():
	# 	try:
	# 		c = get_db().cursor()
	# 		c.execute("SELECT * FROM maps")
	# 		data = c.fetchall()
	# 		c.close()
	# 	except Error as e:
	# 		print(e)
	
	maps_list = maps.read_sql()[1]
	return render_template('navigation.html', title='Navigation',maps = maps_list)



@app.route('/navigation/deletemap',methods=['POST'])
def deletemap():
	mapname = request.get_data().decode('utf-8')
	print(mapname)
	os.system("rm -rf"+" "+os.getcwd()+"/static/maps/"+mapname+".yaml "+os.getcwd()+"/static/maps/"+mapname+".png "+os.getcwd()+"/static/maps/"+mapname+".pgm")

	maps.del_map(mapname)
	# return ("successfully deleted map")
	# flash("successfully deleted map")
	# return redirect(url_for('navigation'))

	# 重新執行資料庫查詢
	maps_list = maps.read_sql()[1]
	return render_template('navigation.html', title='Navigation',maps = maps_list)

	# 強制更新資料來源
	# global maps_list
	# maps_list = maps.read_sql()[1]
	# return "successfully deleted map"




@app.route("/navigation/<variable>" , methods=['GET','POST'])
def gotomapping(variable):
	if variable == "index":
		roslaunch_process.start_mapping()
	elif variable == "gotomapping":		
		roslaunch_process.stop_navigation()
		time.sleep(2)
		roslaunch_process.start_mapping()
	return "success"



@app.route("/navigation/loadmap" , methods=['POST'])
def navigation_properties():

	mapname = request.get_data().decode('utf-8')
	roslaunch_process.stop_navigation()
	time.sleep(1)
	roslaunch_process.start_navigation(mapname)
	return("success")


@app.route("/navigation/stop" , methods=['POST'])
def stop():
	os.system("rostopic pub /move_base/cancel actionlib_msgs/GoalID -- {}") 
	return("stopped the robot")


@app.route('/mapping')
def mapping():
	
	# with get_db():
	# 	try:
	# 		c = get_db().cursor()
	# 		c.execute("SELECT * FROM maps")
	# 		data = c.fetchall()
	# 		c.close()
	# 	except Error as e:
	# 		print(e)
	
	maps_list = maps.read_sql()[1]
	# print(maps_list)
	return render_template('mapping.html', title='Mapping', maps = maps_list) 
	


@app.route("/mapping/cutmapping" , methods=['POST'])
def killnode():
	roslaunch_process.stop_mapping() 
	return("killed the mapping node")



@app.route("/mapping/savemap" , methods=['POST'])
def savemap():
	mapname = request.get_data().decode('utf-8')

	os.system("rosrun map_server map_saver -f"+" "+os.path.join(os.getcwd(),"static/maps/",mapname))
	os.system("convert"+" "+os.getcwd()+"/static/maps/"+mapname+".pgm"+" "+os.getcwd()+"/static/maps/"+mapname+".png")
	# 讀取地圖數據
	with open(os.path.join(os.getcwd(), "static/maps/", mapname+".png"), 'rb') as f:
		mapdata = f.read()

	maps.add_sql([mapname, mapdata])
	
	# with get_db():
	# 	try:
	# 		c = get_db().cursor()
	# 		c.execute("insert into maps (name) values (?)", (mapname,))
	# 		# get_db().commit()
	# 		c.close()
	# 	except Error as e:
	# 		print(e)

    # 讀取地圖數據
	# with open(os.path.join(os.getcwd(), "static/maps/", mapname+".png"), 'rb') as f:
	# 	mapdata = f.read()

	# conn = psycopg2.connect(**DATABASE_URL, sslmode='require')
	# cur = conn.cursor()

	# insert = cur.execute("INSERT INTO maps (name, data) VALUES (%s, %s)")
	# cur.execute(insert, (mapname, psycopg2.Binary(mapdata)))

	# conn.commit()
	# cur.close()
	# conn.close()
		
	return("success")




@app.route("/shutdown" , methods=['POST'])
def shutdown():
	os.system("shutdown now") 
	return("shutting down the robot")	




@app.route("/restart" , methods=['POST'])
def restart():
	os.system("restart now") 
	return("restarting the robot")

# debug
@app.errorhandler(404)
def page_not_found(error):
    return "這個頁面不存在！", 404

@app.errorhandler(Exception)
def handle_exception(error):
    # 可以在這裡記錄錯誤訊息
    return "發生了一個錯誤！", 500



if __name__ == '__main__':
	subprocess.Popen(["roslaunch", "rosbridge_server", "rosbridge_websocket.launch"])
	subprocess.Popen(["rosrun", "robot_pose_publisher", "robot_pose_publisher"])
	
	app.run(host='0.0.0.0', port=8888, debug=True)
	# app.run(host='192.168.1.180', port=8000, debug=False)