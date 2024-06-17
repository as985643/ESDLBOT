# from flask import Flask

# app_test = Flask(__name__)

# @app_test.route('/')
# def hello_world():
#     return 'Hello, World!'

# if __name__ == '__main__':
#     app_test.run(debug=True)

# 导入Flask类
from flask import Flask,render_template,request
import time

# 实例化，可视为固定格式
app_test = Flask(__name__)

# route()方法用于设定路由；类似spring路由配置

@app_test.route('/')
def index():
    return render_template("index.html")
@app_test.route('/login',methods=['POST']) # 设置提交方法为post
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    if username == 'admin' and password=='123456': # 登陆的账号和密码
        login_information='登陆成功'
        return render_template("index.html",login_information=login_information)
    else:
        login_information = '登陆失败'
        return render_template("index.html",login_information=login_information)
if __name__ == '__main__':
    # app_test.run(host, port, debug, options)
    # 默认值：host="127.0.0.1", port=5000, debug=False
    app_test.run(host='0.0.0.0', debug=True, port=87)