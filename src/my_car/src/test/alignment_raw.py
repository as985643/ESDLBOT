#!/usr/bin/env python3
import rospy
from enum import Enum
import tf2_ros
from tf import TransformBroadcaster
from tf.transformations import quaternion_multiply, quaternion_from_euler, euler_from_quaternion, quaternion_matrix
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseActionResult
from math import atan2

class Alignment():
    def __init__(self):
        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.tfbuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfbuffer)
        self.tfbcaster = TransformBroadcaster()
        self.rate = rospy.Rate(10.0)
        self.vel = Twist()
        self.AlignmentSequence = Enum('AlignmentSequence', 'wait_reach_goal searching_code set_target_frame changing_direction moving_nearby facing_code finished')
        self.current_sequence = self.AlignmentSequence.wait_reach_goal.value
        self.target_angle = 0.0
        self.turn_reach = 0
        self.first = True
    
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
            print("Start !")
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
            target_trans, code_rot = self.set_target_frame(0.3)
            while not rospy.is_shutdown():
                self.tfbcaster.sendTransform(target_trans,
                                             code_rot,
                                             rospy.Time.now(),
                                             "target_dummy",
                                             "odom") 
                if self.current_sequence is self.AlignmentSequence.set_target_frame.value:        
                    self.current_sequence = self.AlignmentSequence.changing_direction.value
                
                ########################
                ## Changing direction ##
                ########################
                if self.current_sequence is self.AlignmentSequence.changing_direction.value:
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
                        print("Reach!")
                        self.first = True
                        self.turn_reach = 0
                        self.current_sequence = self.AlignmentSequence.finished.value
                        self.is_sequence_finished = False
                        
                ############
                ## Finish ##
                ############
                elif self.current_sequence is self.AlignmentSequence.finished.value:
                    print("Done !")
                    rospy.signal_shutdown("Done!")
                    
                self.rate.sleep()
            
    def Face_code(self):
        trans = self.listen_tf_frame("base_footprint", "target_dummy")
        base_quat = [trans.transform.rotation.x, 
                     trans.transform.rotation.y,
                     trans.transform.rotation.z,
                     trans.transform.rotation.w]
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
        self.Turning(diff_angle * 1.0)
        if abs(diff_angle) < 0.01:
            self.turn_reach += 1
            if self.turn_reach > 5:
                self.allStop()

                print('1st rotation diff', diff_angle)
                print("rotation done!")
                return True

    def set_target_frame(self, distance=0):
        # find the pose of the virtual code
        code_pose = self.listen_tf_frame("odom", "tag_0")
        
        code_rot = [code_pose.transform.rotation.x,
                    code_pose.transform.rotation.y,
                    code_pose.transform.rotation.z,
                    code_pose.transform.rotation.w]
        code_trans = [code_pose.transform.translation.x,
                      code_pose.transform.translation.y,
                      code_pose.transform.translation.z]
        rot_matrix = quaternion_matrix(code_rot)
        target_offset = rot_matrix.dot([0.0, 0.0, distance, 1.0])
        target_trans = code_trans + target_offset[:3]
        #print(virtual_pos)
        return (target_trans, code_rot)
    
    def Searching_Code(self):
        '''
          Anticlockwise rotation until found AprilTag code.
        '''
        self.Turning(0.2)
        self.listen_tf_frame('usb_cam', 'tag_0')
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
        rospy.sleep(10)
        
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
                return trans
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                #print("Fail", e)
                continue
            self.rate.sleep()
            
    def listen_base_footprint_to_tag(self, tag_name):
        '''
        Listening until get the specify tf frame, then caculate the relative pose
        '''
        trans = self.listen_tf_frame('base_footprint', tag_name)
        trans_distance = [trans.transform.translation.x, trans.transform.translation.y]
        angle = atan2(trans_distance[1], trans_distance[0])
        
        trans = self.listen_tf_frame('odom', 'base_footprint')
        #print("trans", trans)
        base_quat = [trans.transform.rotation.x, 
                     trans.transform.rotation.y,
                     trans.transform.rotation.z,
                     trans.transform.rotation.w]
        x, y, z = euler_from_quaternion(base_quat)
        #print("z", z)
        #print("angle", angle)
        return z + angle
        
    def get_base_yaw(self):
        '''
        Listening until get the specify tf frame, then caculate the relative pose
        '''
        trans = self.listen_tf_frame('odom', 'base_footprint')
        base_quat = [trans.transform.rotation.x, 
                     trans.transform.rotation.y,
                     trans.transform.rotation.z,
                     trans.transform.rotation.w]
        x, y, z = euler_from_quaternion(base_quat)
        #print(z)
        return z      
        
    def listen_deep(self):
        '''
        Listening until get the specify tf frame, then caculate the relative pose
        '''
        trans = self.listen_tf_frame('base_footprint', 'target_dummy')
        deep = trans.transform.translation.x
        #print('deep', deep)
        return deep


#========================================================================================

if __name__ == '__main__':
    rospy.init_node('alignment')
    alignment = Alignment()
    print("Waiting...")
    '''
    while not rospy.is_shutdown():
        alignment.main()
        rospy.sleep(0.1)
    '''
    alignment.listen_tf_frame('usb_cam', 'tag_0')
    rospy.spin()





