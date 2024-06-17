from flask import Flask, render_template, url_for,g,jsonify,request,make_response
import subprocess
import signal
import os
import time
from mySQL import mySQL


app = Flask(__name__)



car_sensor = mySQL("car_sensor")
maps = mySQL("maps")
	
class roslaunch_process():
    @classmethod
    def start_navigation(self,mapname):
        self.process_navigation = subprocess.Popen(["roslaunch","--wait", "my_car", "esdlbot_V1_navigation.launch", "map_file:="+os.getcwd()+"/static/"+mapname+".yaml"])

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
	'''
	with get_db():
	    try:
	        c = get_db().cursor()
	        c.execute("SELECT * FROM maps")
        	data = c.fetchall()
	        c.close()
	    except Error as e:
	        print(e)'''
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

	'''
	with get_db():
	    try:
	        c = get_db().cursor()
	        c.execute("SELECT * FROM maps")
        	data = c.fetchall()
	        c.close()
	    except Error as e:
	        print(e)
	'''
	maps_list = maps.read_sql()[1]
	return render_template('navigation.html', title='Navigation',maps = maps_list)



@app.route('/navigation/deletemap',methods=['POST'])
def deletemap():
	mapname = request.get_data().decode('utf-8')
	print(mapname)
	os.system("rm -rf"+" "+os.getcwd()+"/static/"+mapname+".yaml "+os.getcwd()+"/static/"+mapname+".png "+os.getcwd()+"/static/"+mapname+".pgm")

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
	'''
	with get_db():
	    try:
	        c = get_db().cursor()
	        c.execute("SELECT * FROM maps")
        	data = c.fetchall()
	        c.close()
	    except Error as e:
	        print(e)
	'''
	maps_list = maps.read_sql()[1]
	return render_template('mapping.html', title='Mapping', maps = maps_list) 
	


@app.route("/mapping/cutmapping" , methods=['POST'])
def killnode():
	roslaunch_process.stop_mapping() 
	return("killed the mapping node")



@app.route("/mapping/savemap" , methods=['POST'])
def savemap():
	mapname = request.get_data().decode('utf-8')

	os.system("rosrun map_server map_saver -f"+" "+os.path.join(os.getcwd(),"static",mapname))
	os.system("convert"+" "+os.getcwd()+"/static/"+mapname+".pgm"+" "+os.getcwd()+"/static/"+mapname+".png")
	maps.add_sql([mapname])
	'''
	with get_db():
	    try:
	        c = get_db().cursor()
	        c.execute("insert into maps (name) values (?)", (mapname,))
	        # get_db().commit()
	        c.close()
	    except Error as e:
	        print(e)
    '''
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
	
	
	
	
	
	
	
