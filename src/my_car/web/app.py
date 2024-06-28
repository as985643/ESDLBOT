from flask import Flask, render_template, url_for,g,jsonify,request,make_response,send_from_directory,redirect
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

app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024

car_sensor = mySQL("car_sensor")
maps = mySQL("maps")

DATABASE_URL = {"database":"esdl", "user":"esdl", "password":"bj/6m06",
                "host":"192.168.1.180", "port":"5432"}		# sql 資訊
	
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

@app.route('/AIdetect')
def upload_file():
   return render_template('uploads.html',title='AIdetect')

@app.route('/AIdetect', methods = ['GET', 'POST'])
def uploads():
	if request.method == 'POST':
		f = request.files['file']
		filename = secure_filename(f.filename)
		filepath = os.path.join('./static/uploads', filename)
		f.save(filepath)

		# 在這裡將 '/static/' 從文件路徑中移除，以適應 url_for 函數
		url_for_filepath = filepath.replace('./static/', '')

		# 獲取使用者選擇的模型
		model_choice = request.form.get('model_choice')

		# 根據使用者的選擇來決定要使用哪個模型
		if model_choice == 'Analog_meter_v15':
			model_path = 'static/weights/Analog_meter_v15.pt'
		elif model_choice == 'count_mushrooms':
			model_path = 'static/weights/count_mushrooms.pt'
		elif model_choice == 'digital_meter':
			model_path = 'static/weights/digital_meter.pt'
		else:
			return '未知的模型選擇', 400

		model = YOLO(model_path)

		# result_image = model.predict(filepath, save=True)
		# Run inference on the uploaded image
		result_image = model([filepath], save=True)

		# 將結果轉換為 JSON 格式
		predictions = result_image[0].tojson()
		predictions = json.loads(predictions)
		# print(predictions)

		# 根據 x 座標排序預測結果
		predictions.sort(key=lambda x: (x['box']['x1'], x['box']['y1']))

		# 初始化計數器
		count = 0
		detected_classes = []

		# 遍歷每一個邊界框
		for prediction in predictions:
			# 獲取類別名稱
			class_name = prediction['name']
			detected_classes.append(class_name)
			# 如果這個邊界框的類別是 'King oyster mushroom'
			if class_name == 'King_oyster_mushroom':
				# 增加計數器
				count += 1

		# 如果模型是 count_mushrooms，則計算杏鮑菇的數量
		# if model_choice == 'count_mushrooms':
			# 假設 result 是一個包含所有被辨識出的物件的列表，每個物件都有一個 'class' 鍵和一個 'bbox' 鍵
			# print(result_image[0].boxes)
		# 	mushroom_count = sum(1 for box in result_image[0].boxes if box['class'] == 'King_oyster_mushroom')
		# else:
		# 	mushroom_count = None

		# 將張量轉換為列表
		# predictions = predictions.tolist()

		# 計算King oyster mushroom類別的物件數量
		# mushroom_count = sum(1 for p in predictions if p['class'] == 'King_oyster_mushroom')

		# 獲取所有的預測資料夾
		folders = glob.glob('runs/detect/predict*')

		# 使用資料夾的修改時間來找出最新的資料夾
		latest_folder = max(folders, key=os.path.getmtime)
			
		result_image = os.path.join(latest_folder, filename)

		# 將結果影像移動到static資料夾中
		dst_path = os.path.join('./static/results/', filename)

		# 檢查目標位置是否已經存在同名檔案
		if os.path.exists(dst_path):
			# 如果存在，則先刪除它
			os.remove(dst_path)
		
		shutil.move(result_image, './static/results/')

		# 更新結果影像的路徑
		result_image = os.path.join('results', filename)

		# return '物件偵測完成，結果影像已儲存到: {}'.format(result_image)
		
		# 返回一個HTML模板，並將結果影像的路徑傳遞給該模板
		return render_template('results.html', result_image=result_image, mushroom_count=count,
								original_image=url_for_filepath, detected_classes=detected_classes)

		# ... 其他程式碼不變 ...
	else:
		# 如果是 GET 請求，則顯示上傳表單
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
	return ("successfully deleted map")	




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
	maps.add_sql([mapname])
	
	# with get_db():
	# 	try:
	# 		c = get_db().cursor()
	# 		c.execute("insert into maps (name) values (?)", (mapname,))
	# 		# get_db().commit()
	# 		c.close()
	# 	except Error as e:
	# 		print(e)
    
	return("success")




@app.route("/shutdown" , methods=['POST'])
def shutdown():
	os.system("shutdown now") 
	return("shutting down the robot")	




@app.route("/restart" , methods=['POST'])
def restart():
	os.system("restart now") 
	return("restarting the robot")



if __name__ == '__main__':
	subprocess.Popen(["roslaunch", "rosbridge_server", "rosbridge_websocket.launch"])
	subprocess.Popen(["rosrun", "robot_pose_publisher", "robot_pose_publisher"])
	
	app.run(host='0.0.0.0', port=8000, debug=False)    
	
	
	
	
	
	
	
