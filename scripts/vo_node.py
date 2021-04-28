#!/usr/bin/env python3


import rospy
import numpy as np
from sensor_msgs.msg import Image
import time
import json
import cv2
from std_msgs.msg import String
from tools.PinholeCamera import *

# based on: https://github.com/uoip/monoVO-python


class VisualOdometry(object):
    """
    A simple frame by frame visual odometry
    """

    def __init__(self, cam):
        """
        :param detector: a feature detector can detect keypoints their descriptors
        :param matcher: a keypoints matcher matching keypoints between two frames
        :param cam: camera parameters
        """

        # camera parameters
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)

        # frame index counter
        self.index = 0

        # keypoints and descriptors
        self.kptdescs = {}

        # pose of current frame
        self.cur_R = None
        self.cur_t = None

    def update(self, matches, absolute_scale=0):
        """
        update a new image to visual odometry, and compute the pose
        :param image: input image
        :param absolute_scale: the absolute scale between current frame and last frame
        :return: R and t of current frame
        """
        

        # compute relative R,t between ref and cur frame
        E, mask = cv2.findEssentialMat(matches['cur_keypoints'], matches['ref_keypoints'],
                                        focal=self.focal, pp=self.pp,
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, matches['cur_keypoints'], matches['ref_keypoints'],
                                        focal=self.focal, pp=self.pp)

        print('R:', R)
        print('t:', t)

        # get absolute pose based on absolute_scale
        if (absolute_scale > 0.1):
            self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)

        self.index += 1
        return self.cur_R, self.cur_t


class AbosluteScaleComputer(object):
    def __init__(self):
        self.prev_pose = None
        self.cur_pose = None
        self.count = 0

    def update(self, pose):
        self.cur_pose = pose

        scale = 1.0
        if self.count != 0:
            scale = np.sqrt(
                (self.cur_pose[0, 3] - self.prev_pose[0, 3]) * (self.cur_pose[0, 3] - self.prev_pose[0, 3])
                + (self.cur_pose[1, 3] - self.prev_pose[1, 3]) * (self.cur_pose[1, 3] - self.prev_pose[1, 3])
                + (self.cur_pose[2, 3] - self.prev_pose[2, 3]) * (self.cur_pose[2, 3] - self.prev_pose[2, 3]))

        self.count += 1
        self.prev_pose = self.cur_pose
        return scale



def process_matching_data(data):
    loaded_data = json.loads(data.data)
    data_ = np.array(loaded_data)
    # print(data_.shape)

    matches = {}
    matches['cur_keypoints'] = data_[1]
    matches['ref_keypoints'] = data_[0]
    matches['match_score'] = data_[2]

    R, t = vo.update(matches)
    # print('R:', R)
    # print('t:', t)
    

if __name__ == "__main__":
    rospy.init_node('visual_odometry', anonymous=True)

    # rospy.Subscriber("/stereo/left/image_rect", Image, process_image, queue_size = 1)
    # rospy.Subscriber("/superglue/matches/ref_keypoints", String, process_ref_keypoints, queue_size=50)
    # rospy.Subscriber("/superglue/matches/cur_keypoints", String, process_cur_keypoints, queue_size=50)
    # rospy.Subscriber("/superglue/matches/match_score", String, process_match_score, queue_size=50)

    rospy.Subscriber("/superglue/matches/all_data", String, process_matching_data, queue_size=50)
    path_pub = rospy.Publisher('/superglue/matches/path', Path, queue_size=10)
    
    absscale = AbosluteScaleComputer()

    cam = PinholeCamera(1280.0, 560.0, 816.40221474060002, 817.38388562809996, 608.82658427579997, 266.68865652440002)
    vo = VisualOdometry(cam)

    rospy.spin()

        