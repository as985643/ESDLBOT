#!/usr/bin/env python3
import rospy
from enum import Enum
import tf2_ros
import geometry_msgs.msg
from tf.transformations import quaternion_multiply, quaternion_from_euler, euler_from_quaternion, quaternion_matrix
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseActionResult
from math import atan2
from std_srvs.srv import Trigger, TriggerResponse  # 新增這一行

manual_mode = False

class Alignment():
    def __init__(self):
        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.tfbuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfbuffer)
        self.tfbcaster = tf2_ros.StaticTransformBroadcaster()
        self.rate = rospy.Rate(10.0)
        self.vel = Twist()
        self.AlignmentSequence = Enum('AlignmentSequence', 'wait_reach_goal searching_code set_target_frame changing_direction moving_nearby facing_code finished')
        self.current_sequence = self.AlignmentSequence.wait_reach_goal.value
        self.target_angle = 0.0
        self.turn_reach = 0
        self.first = True
        self.static_transformStamped = geometry_msgs.msg.TransformStamped()

        # 新增一個ROS服務
        self.arm_motion_service = rospy.Service('start_arm_motion', Trigger, self.handle_arm_motion)
    
    def handle_arm_motion(self, req):
        # 當服務被調用時，執行這個函數
        self.main()
        return TriggerResponse(success=True, message="Motion completed")
    
    def wait_manual(self, manual_mode):
        if (manual_mode == True): input("Press Enter to continue")  
        else: return 0
    
    def main(self):
        ###########################
        ## Goal & Searching code ##
        ###########################
        if self.current_sequence is self.AlignmentSequence.wait_reach_goal.value:
            '''
              First : Start the program step by step. 
                      In this program, the user has to manually select a goal for the robot.
                      The program will wait here until the robot reaches the goal.
                      # System will send a message to "move_base/result" Topic when the robot reaches the goal !
            '''
            rospy.wait_for_message("move_base/result", MoveBaseActionResult)
            self.allStop()
            print("Start !")
            self.wait_manual(manual_mode)       # Manual Mode
            self.current_sequence = self.AlignmentSequence.searching_code.value
      
        elif self.current_sequence is self.AlignmentSequence.searching_code.value: 
            '''
              Second : In this section, it is indicated that the robot has reached the goal, 
                       but we still need to check if the AprilTag is present in the camera frame. 
                       Therefore rotate the robot until system get the transform between robot and AprilTag.
            '''  
            self.is_sequence_finished = self.Searching_Code()
            if self.is_sequence_finished is True:
                print("find code !")
                self.wait_manual(manual_mode)       # Manual Mode
                self.current_sequence = self.AlignmentSequence.set_target_frame.value
                self.is_sequence_finished = False
                
        ############################
        ## Set dummy target frame ##
        ############################    
        elif self.current_sequence is self.AlignmentSequence.set_target_frame.value:
            '''
              Third : Now, we have confirmed that the robot has reached the goal and found the AprilTag code,
                      but the code may disapear when the robot moves, so set up a dummy target frame 
                      to point to the "odom" frame to advoid this problem.
            '''  
            print("Setting dummy target frame...") 
            rospy.sleep(2)
            self.set_target_frame("tag_1", 0.55)      # Set so that the dummy target frame is 0.3 m front of the AprilTag code
            
            self.tfbcaster.sendTransform(self.static_transformStamped)      # Broadcaster the dummy target frame
            
            self.wait_manual(manual_mode)       # Manual Mode      
            self.current_sequence = self.AlignmentSequence.changing_direction.value
            
        ########################
        ## Changing direction ##
        ########################
        elif self.current_sequence is self.AlignmentSequence.changing_direction.value:
            if self.first is True:
                print("changing_direction...")
                self.first = False
            # get the target angle
            self.target_angle = self.listen_base_footprint_to_tag("target_dummy")
            #print("need", angle)
            self.is_sequence_finished = self.ChangingDirection()
            
            if self.is_sequence_finished is True:
                print("Reach!")
                self.first = True
                self.turn_reach = 0
                self.wait_manual(manual_mode)       # Manual Mode
                self.current_sequence = self.AlignmentSequence.moving_nearby.value
                self.is_sequence_finished = False
                        
        ###################
        ## Moving nearby ##
        ###################
        elif self.current_sequence is self.AlignmentSequence.moving_nearby.value:
            if self.first is True:
                print("moving...")
                self.first = False
            self.is_sequence_finished = self.Move_nearby()
           
            if self.is_sequence_finished is True:
                print("Reach!")
                self.first = True
                self.wait_manual(manual_mode)       # Manual Mode
                self.current_sequence = self.AlignmentSequence.facing_code.value
                self.is_sequence_finished = False
                        
        #################
        ## Facing code ##
        #################
        elif self.current_sequence is self.AlignmentSequence.facing_code.value:
            if self.first is True:
                print("Facing to code...")
                self.first = False
            self.is_sequence_finished = self.Face_code()
            
            if self.is_sequence_finished is True:
                self.allStop()
                print("Reach!")
                self.first = True
                self.turn_reach = 0
                self.wait_manual(manual_mode)       # Manual Mode
                self.current_sequence = self.AlignmentSequence.finished.value
                self.is_sequence_finished = False
                      
        ############
        ## Finish ##
        ############
        elif self.current_sequence is self.AlignmentSequence.finished.value:
            print("Done !")
            rospy.signal_shutdown("Done!")

            
    def Face_code(self):
        trans = self.listen_tf_frame("base_footprint", "target_dummy")
        base_quat = [trans.rotation.x, 
                     trans.rotation.y,
                     trans.rotation.z,
                     trans.rotation.w]
        rot = quaternion_multiply(base_quat, [-0.5, 0.5, 0.5, 0.5 ])
        x, y, z = euler_from_quaternion(rot)
        #print("z", z)
        self.Turning(z * 1.0)
        if abs(z) < 0.01:
            self.turn_reach += 1
            if self.turn_reach > 5:
                self.allStop()
                #deep = listen_deep("tag_0")
                #print("target_deep", deep)
                print('2st rotation diff', z)
                return True                   

    def Move_nearby(self):
        deep = self.listen_deep()
        if deep < 0.2:
            self.GoStraight(deep*0.5)
        else:
            self.GoStraight(0.1)

        if deep < 0.0015:
            self.allStop()
            print('position diff', deep)
            return True
    
    def ChangingDirection(self):
        base_angle = self.get_base_yaw()
        diff_angle = self.target_angle - base_angle
        #print("diff_angle", diff_angle)
        self.Turning(diff_angle * 3.0)
        if abs(diff_angle) < 0.05:
            self.turn_reach += 1
            if self.turn_reach > 5:
                self.allStop()

                print('1st rotation diff', diff_angle)
                print("rotation done!")
                return True

    def set_target_frame(self, target_frame, distance=0.0):
        '''
          This function is for caculate the dummy target frame.
          
        '''
        # find the pose of the virtual code
        code_trans = self.listen_tf_frame("odom", target_frame)
        code_pos = [code_trans.translation.x, code_trans.translation.y, code_trans.translation.z]
        code_rot = [code_trans.rotation.x, code_trans.rotation.y, code_trans.rotation.z, code_trans.rotation.w]
        
        rot_matrix = quaternion_matrix(code_rot)
        target_offset = rot_matrix.dot([0.0, 0.0, distance, 1.0])
        target_pos = code_pos + target_offset[:3]
   
        self.static_transformStamped.header.stamp = rospy.Time.now()
        self.static_transformStamped.header.frame_id = "odom"
        self.static_transformStamped.child_frame_id = "target_dummy"
        self.static_transformStamped.transform.translation.x = target_pos[0]
        self.static_transformStamped.transform.translation.y = target_pos[1]
        self.static_transformStamped.transform.translation.z = target_pos[2]
   
        self.static_transformStamped.transform.rotation.x = code_rot[0]
        self.static_transformStamped.transform.rotation.y = code_rot[1]
        self.static_transformStamped.transform.rotation.z = code_rot[2]
        self.static_transformStamped.transform.rotation.w = code_rot[3]

        return 0
    
    def Searching_Code(self):
        '''
          Anticlockwise rotation until found AprilTag code.
        '''
        self.Turning(0.2)
        self.listen_tf_frame('usb_cam', 'tag_1')
        self.allStop()
        return True
    
    def allStop(self):
        self.vel.angular.x = 0.0
        self.vel.angular.y = 0.0
        self.vel.angular.z = 0.0
        self.vel.linear.x = 0.0
        self.vel.linear.y = 0.0
        self.vel.linear.z = 0.0
        self.cmd_pub.publish(self.vel)
        rospy.sleep(0.1)
        self.cmd_pub.publish(self.vel)
        rospy.sleep(0.1)
        
    def Turning(self, z):
        self.vel.angular.x = 0.0
        self.vel.angular.y = 0.0
        self.vel.angular.z = z
        self.vel.linear.x = 0.0
        self.vel.linear.y = 0.0
        self.vel.linear.z = 0.0
        self.cmd_pub.publish(self.vel)
        
    def GoStraight(self, x):
        self.vel.angular.x = 0.0
        self.vel.angular.y = 0.0
        self.vel.angular.z = 0.0
        self.vel.linear.x = x
        self.vel.linear.y = 0.0
        self.vel.linear.z = 0.0
        self.cmd_pub.publish(self.vel)
        
    def listen_tf_frame(self, parent_frame, child_frame):
        '''
          Get and return the relative transform of the input frames 
          which involve translation(x, y, z) and rotation(w, x, y, z) .
          
          # Program will stuck here until get the relative transform ! 
        '''
        while not rospy.is_shutdown():
            try:
                trans = self.tfbuffer.lookup_transform(parent_frame, child_frame, rospy.Time())
                #print(trans, "\n===")
                return trans.transform
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                #print("Fail", e)
                continue
            self.rate.sleep()
            
    def listen_base_footprint_to_tag(self, tag_name):
        '''
        Listening until get the specify tf frame, then caculate the relative pose
        '''
        trans = self.listen_tf_frame('base_footprint', tag_name)
        trans_distance = [trans.translation.x, trans.translation.y]
        angle = atan2(trans_distance[1], trans_distance[0])
        
        trans = self.listen_tf_frame('odom', 'base_footprint')
        #print("trans", trans)
        base_quat = [trans.rotation.x, 
                     trans.rotation.y,
                     trans.rotation.z,
                     trans.rotation.w]
        x, y, z = euler_from_quaternion(base_quat)
        #print("z", z)
        #print("angle", angle)
        return z + angle
        
    def get_base_yaw(self):
        '''
        Listening until get the specify tf frame, then caculate the relative pose
        '''
        trans = self.listen_tf_frame('odom', 'base_footprint')
        base_quat = [trans.rotation.x, 
                     trans.rotation.y,
                     trans.rotation.z,
                     trans.rotation.w]
        x, y, z = euler_from_quaternion(base_quat)
        #print(z)
        return z      
        
    def listen_deep(self):
        '''
        Listening until get the specify tf frame, then caculate the relative pose
        '''
        trans = self.listen_tf_frame('base_footprint', 'target_dummy')
        deep = trans.translation.x
        #print('deep', deep)
        return deep


#========================================================================================

if __name__ == '__main__':
    rospy.init_node('alignment_alter')
    alignment = Alignment()
    print("Waiting...")
    
    while not rospy.is_shutdown():
        alignment.main()
        rospy.sleep(0.1)
    
    #alignment.listen_tf_frame('usb_cam', 'tag_0')
    #alignment.listen_tf_frame('tag_0', 'usb_cam')
    rospy.spin()
    





