import paramiko
import warnings

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect('192.168.1.176', username='ubuntu', password='aibot1234')

    stdin, stdout, stderr = ssh.exec_command("rosrun mycobot_startup_pose startup_pose.py")
finally:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssh.close()
