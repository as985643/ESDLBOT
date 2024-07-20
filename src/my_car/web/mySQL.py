import os
import psycopg2

# DATABASE_URL = {"database":"esdldb", "user":"esdl", "password":"bj/6m06",
#                "host":"localhost", "port":"5432"}		# sql 資訊

# DATABASE_URL = {"database":"mydb", "user":"postgres", "password":"bj/6m06",
#                 "host":"192.168.1.179", "port":"5432"}		# sql 資訊

DATABASE_URL = {"database":"esdl", "user":"esdl", "password":"bj/6m06",
                "host":"192.168.1.180", "port":"5432"}		# sql 資訊

# DATABASE_URL = {"database":"esdl", "user":"esdl", "password":"bj/6m06",
#                 "host":"192.168.67.102", "port":"5432"}		# sql 資訊


class mySQL:
	def __init__(self, table_name):
		self.table_name = table_name

	# 建立 sql 表單
	def creat_sql(self, table_items):
		#
		# table_item must be => ("ItemA", "ItemB", "ItemC")
		#
		# Enter the needed items, then system will ask the type for each item,
		# only to do is type 1, 2 or 3 then enter, if success system will say
		# "The table has creat successfully !"
		try:
			conn = psycopg2.connect(**DATABASE_URL, sslmode='require')							# 連接 Postgresql
			cursor = conn.cursor()
			
			type_dict = {"1":"text", "2":"real", "3":"integer"}
			
			items_msg = ""
			
			for item in table_items:
				type_no = input(f'''Input the type of {item} in number. (1:text, 2:real, 3:integer) : ''')
				items_msg += f''', {item} {type_dict[type_no]}'''

			cursor.execute(f'''CREATE TABLE {self.table_name}({items_msg[2:]});''')							# 執行 SQL
			conn.commit()
			print("The table has creat successfully !")
		except Exception as e:
			print(e)
			print("The table name isn't exist !")
			return "error"
		finally:
			cursor.close()
			conn.close()

	# 刪除 sql 表單
	def bye_sql(self):
		# This will delete all table
		# You need to be careful, the exsist table will not reappear after excute this command !
		try:
			conn = psycopg2.connect(**DATABASE_URL, sslmode='require')							# 連接 Postgresql
			cursor = conn.cursor()
			cursor.execute(f'''DROP TABLE IF EXISTS {self.table_name}''')							# 執行 SQL
			conn.commit()
			print(f"""The table "{self.table_name}" was murdered by you !!!""")
		except Exception as e:
			print(e)
			print("The table doesn't delete !")
		finally:
			cursor.close()
			conn.close()

	# Change type of sql 表單欄位
	def retype_sql_column(self, column_name):
		# You can use this command to modify the type of the exist item,
		# the method like create table, system will ask the needed type,
		# only do is type 1, 2 or 3 then enter.  
		try:
			conn = psycopg2.connect(**DATABASE_URL, sslmode='require')							# 連接 Postgresql
			cursor = conn.cursor()
			type_dict = {"1":"text", "2":"real", "3":"integer"}
			type_no = input(f'''Input the type of {column_name} in number. (1:text, 2:real, 3:integer) : ''')
			cursor.execute(f'''ALTER TABLE {self.table_name} ALTER COLUMN {column_name} TYPE {type_dict[type_no]};''')		# 執行 SQL
			conn.commit()
			print(f"""The column {column_name} retype successfully !!!""")
		except Exception as e:
			print(e)
			print("The column doesn't retype !")
		finally:
			cursor.close()
			conn.close()
	
	# 新增 sql 表單欄位
	def new_sql_column(self, column_name):
		# Use this command to add a new item, the new item will add in the rightmost,
		# system will also ask the type of the item, type 1, 2 or 3 then enter. 
		try:
			conn = psycopg2.connect(**DATABASE_URL, sslmode='require')							# 連接 Postgresql
			cursor = conn.cursor()
			type_dict = {"1":"text", "2":"real", "3":"integer"}
			type_no = input(f'''Input the type of {column_name} in number. (1:text, 2:real, 3:integer) : ''')
			cursor.execute(f'''ALTER TABLE {self.table_name} ADD COLUMN {column_name} {type_dict[type_no]};''')		# 執行 SQL
			conn.commit()
			print(f"""The column "{column_name}" add successfully !!!""")
		except Exception as e:
			print(e)
			print("The column doesn't add !")
		finally:
			cursor.close()
			conn.close()
			
	# 刪除 sql 表單欄位
	def del_sql_column(self, column_name):
		# Use this command to delete an exist item, the data under the item will also disapear.
		try:
			conn = psycopg2.connect(**DATABASE_URL, sslmode='require')							# 連接 Postgresql
			cursor = conn.cursor()
			cursor.execute(f'''ALTER TABLE {self.table_name} DROP COLUMN {column_name};''')		# 執行 SQL
			conn.commit()
			print(f"""Bye~ "{column_name}"!!!""")
		except Exception as e:
			print(e)
			print("The column doesn't delete !")
		finally:
			cursor.close()
			conn.close()


	# 變更 sql 欄位名稱
	def rename_sql_column(self, old_name, new_name):
		try:
			conn = psycopg2.connect(**DATABASE_URL, sslmode='require')							# 連接 Postgresql
			cursor = conn.cursor()
			cursor.execute(f'''ALTER TABLE {self.table_name} RENAME COLUMN {old_name} TO {new_name};''')			# 執行 SQL
			conn.commit()
			print(f"""The column name has been changed !!!""")
		except Exception as e:
			print(e)
			print("The column name doesn't change !")
		finally:
			cursor.close()
			conn.close()

	# 讀取 sql 中每一列之值 並以 list 返回
	def read_sql(self):
		try:
			conn = psycopg2.connect(**DATABASE_URL, sslmode='require')							# 連接 Postgresql
			cursor = conn.cursor()
		
			cursor.execute(f"""SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{self.table_name}';""")
			columns = cursor.fetchall()
			colname = [column[0] for column in columns]
			coltype = [column[1] for column in columns]
			colnamet = colname.copy()
			for i in range(len(colname)):
				if len(colname[i]) < 8:
					colnamet[i] = colname[i] + '\t'
			#print(*colnamet, sep='\t')
			#print(*coltype, sep='\t\t')
			#print("=====================")
			cursor.execute(f"""select * from {self.table_name} ORDER BY record_no ASC""")
			#row_cnt = cursor.rowcount															# 計算總列數
			#row = cursor.fetchmany(row_cnt1)														# 所有列值以 list 紀錄
			row = cursor.fetchall()
			#print(row, "\n=============================")
			return colname, row
		except Exception as e:
			print(e)
			print("The table name isn't exist !")
			return "error"
		finally:
			cursor.close()
			conn.close()

	# 新增資料至 sql 中
	def add_sql(self, add_items):
		#########################################################
		# table_columns = (ItemA, ItemB, ItemC) 		  #
		# add_items = ["18:00", "這則訊息永遠不會出現"]	  #
		#########################################################
		try:
			table_list = self.read_sql()
			print(table_list)
			table_list[1][0]
			lastColumns = table_list[1][-1]												# 取得最後一列資料
		except Exception as e:
			lastColumns = [0]
		try:
			conn = psycopg2.connect(**DATABASE_URL, sslmode='require')							# 連接 Postgresql
			cursor = conn.cursor()
			add_items.insert(0, lastColumns[0] + 1)												# 將 record_no 加入列表
			values = ",".join(["%s"] * len(add_items))
			table_items_txt = "(" + ", ".join(table_list[0]) + ")"
			postgres_insert_query = f"""INSERT INTO {self.table_name} {table_items_txt} VALUES ({values})"""
			cursor.execute(postgres_insert_query, add_items)
			conn.commit()
			count = cursor.rowcount
			print(count, "Record inserted successfully into table")
		except Exception as e:
			print(e)
			print("Record inserted error")
			return "error"
		finally:
			cursor.close()
			conn.close()

	# 更新 sql 資料
	def update_sql(self, item, no, value):
		# Modify the data which record_no number is no to value in the item.
		try:
			conn = psycopg2.connect(**DATABASE_URL, sslmode='require')							# 連接 Postgresql
			cursor = conn.cursor()
			sql_update_query = f"""Update {self.table_name} set {item} = {value} where record_no = {no}"""
			cursor.execute(sql_update_query)
			conn.commit()
			count = cursor.rowcount
			print(count, "Record Update successfully into table")
		except Exception as e:
			print("Record update error")
			return "error"
		finally:
			cursor.close()
			conn.close()

	# 刪除 sql 中指定一列
	def del_sql(self, no):
		# Delete the data which record_no number is no.
		try: 
			conn = psycopg2.connect(**DATABASE_URL, sslmode='require')							# 連接 Postgresql
			cursor = conn.cursor()
			cursor.execute(f"""DELETE FROM {self.table_name} WHERE record_no = {no}""")
			conn.commit()
			count = cursor.rowcount
			print(count, "Record successfully delete from table")
			# 將序號重新編排
			row_no = [x[0] for x in self.read_sql()[1]]										# 取得舊序號
			#print(row_no)
			# 取代舊序號
			for i in range(1, len(row_no)+1):
				sql_update_query = f"""Update {self.table_name} set record_no = %s where record_no = %s"""
				cursor.execute(sql_update_query, (i, row_no[i-1]))
				conn.commit()
			print("序號重新編排成功 !")
		except Exception as e:
			print(e)
			return "error"
		finally:
			cursor.close()
			conn.close()
	
	def del_map(self, name):
		# Delete the data which names is name.
		try: 
			conn = psycopg2.connect(**DATABASE_URL, sslmode='require')							# 連接 Postgresql
			cursor = conn.cursor()
			cursor.execute(f"""DELETE FROM {self.table_name} WHERE column_name = '{name}'""")
			conn.commit()
			count = cursor.rowcount
			print(count, "Record successfully delete from table")
			# 將序號重新編排
			row_no = [x[0] for x in self.read_sql()[1]]										# 取得舊序號
			#print(row_no)
			# 取代舊序號
			for i in range(1, len(row_no)+1):
				sql_update_query = f"""Update {self.table_name} set record_no = %s where record_no = %s"""
				cursor.execute(sql_update_query, (i, row_no[i-1]))
				conn.commit()
			print("序號重新編排成功 !")
		except Exception as e:
			print(e)
			return "error"
		finally:
			cursor.close()
			conn.close()

#### test ######## test  ######## test ######## test ######## test ######## test ######## test ########
if __name__ == '__main__':
	def main():
		'''
		sche = 'Schedule'
		columns = '(record_no, sche_time, sche_work)'
		test_columns = '(record_no, sche_time, sche_work)'
		dicc = {"record_no":"integer", "sche_time":"text", "sche_work":"text"}
		aaa = {"record_no":"integer", "ID":"text", "total":"integer", "fifty":"integer", "ten":"integer", "five":"integer", "one":"integer"}
		'''
		###################################################################################################
		# 手動更新area資料
		# try:
		# 	conn = psycopg2.connect(**DATABASE_URL, sslmode='require') # 連接 Postgresql
		# 	cursor = conn.cursor()
		# 	cursor.execute("UPDATE car_sensor SET area = 'A1' WHERE area = '0';") # 執行 SQL
		# 	conn.commit()
		# 	print("Record updated successfully")
		# except Exception as e:
		# 	print("Error updating record:", e)
		# finally:
		# 	cursor.close()
		# 	conn.close()
		###################################################################################################

		###################################################################################################
		# 手動新增資料
		# data = [
		# 	[1, 'A2', 6, 7, 7.3, '2022-11-3 11:59:01'],
		# 	[2, 'A1', 10, 11, 6.5, '2022-11-3 11:59:01'],
		# 	[3, 'A3', 6, 7, 4.8, '2022-11-3 11:59:01'],
		# 	[4, 'A2', 6, 7, 2.5, '2022-11-3 11:59:01'],
		# 	[5, 'A1', 10, 11, 3.3, '2022-11-3 11:59:01'],
		# 	[6, 'A3', 6, 7, 5.5, '2022-11-3 11:59:01'],
		# 	[7, 'A2', 10, 11, 8.9, '2022-11-3 11:59:01'],
		# 	[8, 'A1', 6, 7, 4.1, '2022-11-3 11:59:01'],
		# 	[9, 'A3', 10, 11, 5.6, '2022-11-3 11:59:01'],
		# 	[10, 'A3', 10, 11, 1.2, '2022-11-3 11:59:01'],
		# 	[11, 'A1', 6, 7, 8.8, '2022-11-3 11:59:01'],
		# 	[12, 'A2', 10, 11, 6.5, '2022-11-3 11:59:01'],
		# 	[13, 'A4', 14, 15, 7.2, '2022-11-3 11:59:01']
		# ]

		# try:
		# 	conn = psycopg2.connect(**DATABASE_URL, sslmode='require') # 連接 Postgresql
		# 	cursor = conn.cursor()

		# 	for row in data:
		# 		cursor.execute(
		# 			"INSERT INTO car_sensor VALUES (%s, %s, %s, %s, %s, %s);",
		# 			row
		# 		)

		# 	conn.commit()
		# 	print("Records inserted successfully")
		# except Exception as e:
		# 	print("Error inserting records:", e)
		# finally:
		# 	cursor.close()
		# 	conn.close()
		###################################################################################################

		###################################################################################################
		# 刪除 car_sensor 表格中的所有資料
		# try:
		# 	conn = psycopg2.connect(**DATABASE_URL, sslmode='require') # 連接 Postgresql
		# 	cursor = conn.cursor()

		# 	cursor.execute("DELETE FROM car_sensor;") # 執行 SQL
		# 	conn.commit()
		# 	print("All data deleted successfully from the table")
		# except Exception as e:
		# 	print("Error deleting data:", e)
		# finally:
		# 	cursor.close()
		# 	conn.close()
		###################################################################################################
		
		sql = mySQL("car_sensor")
		# sql = mySQL("maps")
		
		## Delete specify row ##
		#for i in range(100):
			#sql.del_sql(1)
		
		## Create new table ##
		#table_item = ("record_no", "names")
		#sql.creat_sql(table_item)
		
		## Add new data ##
		#sql.add_sql(msg)
		
		## Delete table ##
		#sql.bye_sql()
		
		## Retype the type of item ##
		#sql.retype_sql_column('area')
		
		## Add new item ##
		#sql.new_sql_column("Humidity")	
		#sql.new_sql_column("CO2")	
		#sql.new_sql_column("Time")
			
		## Delete item ##
		#sql.del_sql_column("ItemB")
		#sql.del_map("test")
		
		## Change item name ##
		#sql.rename_sql_column("demo", "Area")
		#sql.rename_sql_column("demo2", "Temperture")
		
		## Udate values ##
		#sql.update_sql("demo", 1, 3)
		
		## Read all table ##
		print(sql.read_sql())
		#import time
		#conn = psycopg2.connect(**DATABASE_URL, sslmode='require')							# 連接 Postgresql
		#cursor = conn.cursor()
		#while 1:
		#	cursor.execute(f'''Select * FROM car_sensor LIMIT 0''')
		#	colnames = [desc[0] for desc in cursor.description]
		#	print(colnames)										# 所有列值以 list 紀錄
		#	time.sleep(1)


#### test ######## test  ######## test ######## test ######## test ######## test ######## test ########

	main()
