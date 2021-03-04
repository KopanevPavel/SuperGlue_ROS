#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Float32
import message_filters
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def process_image(msg):
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(msg)
    # print(img.shape)
    # cv2.imwrite('test.jpg', img)

    

if __name__ == '__main__':
    while not rospy.is_shutdown():
        rospy.init_node('superglue', anonymous=True)

        #rospy.Subscriber("/stereo/left/image_raw", Image, process_image)
        rospy.Subscriber("/stereo/left/image_rect", Image, process_image)
        

        rospy.spin()