#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64

def talker():
    pub = rospy.Publisher('/elevating_pos', Float64, queue_size=10)
    rospy.init_node('talker', anonymous=True)

    while not rospy.is_shutdown():
        pos_msg = Float64()
        pos_msg.data = float(input("請輸入升降位置值: ")) # 從使用者輸入獲取升降位置值
        rospy.loginfo(pos_msg)
        pub.publish(pos_msg)

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
