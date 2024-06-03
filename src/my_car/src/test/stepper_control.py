#!/usr/bin/env python3
import RPi.GPIO as GPIO
from geometry_msgs.msg import Twist
import math
import threading
import time

wheel_diameter = rospy.get_param("wheel_diameter", 0.067)
Pulse_Per_Rev = rospy.get_param("Pulse_Per_Rev", 6400)

class stepper:
    def __init__(self):
        self.LPulsePin = 17
        self.RPulsePin = ?
        self.LDirPin = ?
        self.RDirPin = ?
        self.LInterruptPin = ?
        self.RInterruptPin = ?
        self.Lencoder = 0
        self.Rencoder = 0
        self.rate = 0
        self.gpioSetup()
        self.past_time = rospy.get_time()
        self.real_vel = Twist()
        
        self.realvel_pub = rospy.Publisher('real_vel', Twist, queue_size=10)
        rospy.Subscriber('cmd_vel', Twist, self.subCB)
        self.encoderThread = threading.Thread(target = self.encoder)
        self.jobThread = threading.Thread(target = self.job)
                
    def gpioSetup(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        
        # Set pulse pin mode
        GPIO.setup(self.LPulsePin , GPIO.OUT)
        GPIO.setup(self.RPulsePin , GPIO.OUT)
        # Set direction pin mode
        GPIO.setup(self.LDirPin , GPIO.OUT)
        GPIO.setup(self.RDirPin , GPIO.OUT)
        # Set interrupt pin mode
        GPIO.setup(self.LInterruptPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.RInterruptPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        ## Interrupt ##
        GPIO.add_event_detect(self.LInterruptPin, GPIO.RISING, callback=self.LInterruptCB)
        GPIO.add_event_detect(self.RInterruptPin, GPIO.RISING, callback=self.RInterruptCB)
    
    def LInterruptCB(self, channel):
        self.Lencoder += 1
        
    def RInterruptCB(self, channel):
        self.Rencoder += 1
        
    def subCB(self, vel):
        v = vel.linear.x
        self.rate = (6400*v) / (wheel_diameter*math.pi) if v != 0 else 0
    
    def halt(self):
        GPIO.output(self.LDirPin , 0)
        GPIO.output(self.RDirPin , 0)
        	
    def forward(self):
        GPIO.output(self.LDirPin , 1)
        GPIO.output(self.RDirPin , 1)
        try:
            GPIO.output(self.LPulsePin , 0)
            GPIO.output(self.RPulsePin , 1)
            time.sleep(1/self.rate)
            GPIO.output(self.LPulsePin , 0)
            GPIO.output(self.RPulsePin , 0)
            time.sleep(1/self.rate)
        except:
            break
    
    def backward(self):
        GPIO.output(self.LDirPin , 1)
        GPIO.output(self.RDirPin , 0)
        try:
            GPIO.output(self.LPulsePin , 1)
            GPIO.output(self.RPulsePin , 1)
            rospy.sleep(-1/self.rate)
            GPIO.output(self.LPulsePin , 0)
            GPIO.output(self.RPulsePin , 0)
            rospy.sleep(-1/self.rate)
        except:
            break
        
    def encoder(self):
        while not rospy.is_shutdown():
            t = rospy.get_time()
            dt = t - self.past_time 
            if dt >= 0.01:
                Lvelocity = self.Lencoder / Pulse_Per_Rev * wheel_diameter*math.pi / dt
                Rvelocity = self.Rencoder / Pulse_Per_Rev * wheel_diameter*math.pi / dt
                self.real_vel.linear.x = Lvelocity
                self.realvel_pub.publish(self.real_vel)
                self.Lencoder = 0
                self.Rencoder = 0
                self.past_time = rospy.get_time()
            
    def job(self):
        while not rospy.is_shutdown():
            if self.rate == 0: self.halt()
            elif self.rate > 0: self.forward()
            elif self.rate < 0: self.backward()

def signal_handler(sig, frame):
    GPIO.cleanup()
    rospy.signal_shutdown("Done!")
    sys.exit(0)  
      	
if __name__ == "__main__":
    
    
    rospy.init_node('motor_control', anonymous=True)
    
    motor = stepper()
    
	motor.encoderThread.start()
    motor.jobThread.start() 
	
    rospy.spin()
    
    motor.encoderThread.join()
    motor.jobThread.join() 
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
