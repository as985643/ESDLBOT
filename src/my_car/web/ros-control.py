import websocket
import json

def on_message(ws, message):
    print("Received message: " + message)

def on_error(ws, error):
    print("Error: " + str(error))

def on_close(ws):
    print("Connection closed")

def on_open(ws):
    print("Connection opened")
    # 你可以在這裡發送消息到ROS節點
    # 例如，以下的代碼會運行你的ROS節點
    message = {
        'op': 'call_service',
        'service': '/rosapi/service_type',
        'args': {
            'service': '/mycobot_startup_pose/startup_pose'
        }
    }
    ws.send(json.dumps(message))

if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://192.168.1.176:9090",  # 將此IP地址更改為你的Rosbridge服務器的IP地址
                              on_message = on_message,
                              on_error = on_error,
                              on_close = on_close)
    ws.on_open = on_open
    ws.run_forever()
