from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('mapping.html')

@app.route('/save_data', methods=['POST'])
def save_data():
    data = request.get_data().decode('utf-8')
    # 在這裡處理和保存數據
    return "Data received: " + data

if __name__ == '__main__':
    app.run(debug=True)