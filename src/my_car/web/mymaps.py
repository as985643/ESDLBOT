import psycopg2
from PIL import Image
import numpy as np
import base64
from mySQL import mySQL

maps = mySQL("ros_maps")

# Load the PGM file
img = Image.open('/home/esdl/feng_ws/src/my_car/web/static/maps/lab2.pgm')

# Convert the image to a NumPy array
img_array = np.array(img)

# Convert the NumPy array to bytes
img_bytes = img_array.tobytes()

# Encode the bytes to a base64 string
img_str = base64.b64encode(img_bytes).decode('utf-8')

# Connect to the PostgreSQL database
conn = psycopg2.connect(database="esdl", user="esdl", password="bj/6m06", host="192.168.1.180", port="5432")

# Create a cursor object
cur = conn.cursor()

# Insert the map data into the database
cur.execute("INSERT INTO maps (record_no, column_name, data_type) VALUES (%s, %s, %s)", ('5', 'lab2', img_str))
# cur.execute("INSERT INTO maps (column_name, data_type) VALUES (%s, %s)", ('lab2', img_str))
# maps_list = maps.read_sql()[1]
# print(maps_list)

# cur.execute("SELECT * FROM maps")

# rows = cur.fetchall()

# for row in rows:
#     print(row)

# Commit the changes and close the connection
conn.commit()
cur.close()
conn.close()
