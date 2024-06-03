#!/usr/bin/env python3  
#import roslib
#roslib.load_manifest('learning_tf')

import rospy
import tf

if __name__ == '__main__':
    rospy.init_node('fixed_tf_broadcaster')
    std = tf.TransformBroadcaster()
    code = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        std.sendTransform((0.0, 0.0, 0.30),
                         (-0.00415626, -0.00543869, -0.19533397, 0.9807129),
                         rospy.Time.now(),
                         "fake_tag_0",
                         "odom")
        '''
        code.sendTransform((0.5, 0.0, 0.5),
                         (-0.36653816,  0.59395848,  0.60829264, -0.37794598),
                         rospy.Time.now(),
                         "code",
                         "base")
                         
        code.sendTransform((0.5, 0.5, 0.5),
                         (0.5, -0.5, -0.5, 0.5),
                         rospy.Time.now(),
                         "paper",
                         "base")
        '''
        rate.sleep()
