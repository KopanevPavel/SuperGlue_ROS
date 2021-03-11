#!/usr/bin/env python3

import rospy
from std_msgs.msg import String, Float32
import message_filters
import cv2
#from cv_bridge import CvBridge
import numpy as np
from tools.tools import *
from sensor_msgs.msg import Image
import time

prev_time = None
time_list = []

def process_image(msg):
    #bridge = CvBridge()
    #img = bridge.imgmsg_to_cv2(msg)
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

    global prev_time
    global time_list
    
    if prev_time is None:
        prev_time = time.time()
        time_list = []
    else:
        cur_time = time.time()
        delta = cur_time - prev_time
        time_list += [delta]
        prev_time = time.time()

        print('Time:', delta)
        print('Avr time:', np.mean(time_list))


    #print(img.shape)
    kptdescs = {}
    imgs = {}

    img = np.squeeze(img)
    kptdescs = detector(img)

    img = plot_keypoints(img, kptdescs["keypoints"], kptdescs["scores"])
    cv2.imwrite('detector.jpg', img)
    #cv2.imshow("SuperPoint", img)
    #cv2.waitKey()

    

if __name__ == '__main__':

    from detector import SuperPointDetector

    detector = SuperPointDetector({"cuda": 0})

    rospy.init_node('superpoint_detector', anonymous=True)

    #rospy.Subscriber("/stereo/left/image_raw", Image, process_image)
    rospy.Subscriber("/stereo/left/image_rect", Image, process_image, queue_size = 1)


    rospy.spin()
