#!/usr/bin/env python3

import os.path as path
from tools.tools import *
import logging
from models.superglue import SuperGlue
import rospy
import numpy as np
from sensor_msgs.msg import Image
import time
import json
from std_msgs.msg import String
from collections import deque 
from threading import Lock
from detector import SuperPointDetector
from tqdm import tqdm


class SuperGlueMatcher(object):
    default_config = {
        "descriptor_dim": 256,
        "weights": "outdoor",
        "keypoint_encoder": [32, 64, 128, 256],
        "GNN_layers": ["self", "cross"] * 9,
        "sinkhorn_iterations": 100,
        "match_threshold": 0.2,
        "cuda": True
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        #self.config = merge_two_dicts(self.config, config)
        #print(self.config)
        logging.info("SuperGlue matcher config: ")
        logging.info(self.config)

        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'

        assert self.config['weights'] in ['indoor', 'outdoor']
        path_ = path.abspath(path.join(__file__ ,"../..")) + '/models/SuperGluePretrainedNetwork/models/weights/superglue_{}.pth'.format(self.config['weights'])
        self.config["path"] = path_

        logging.info("creating SuperGlue matcher...")
        self.superglue = SuperGlue(self.config).to(self.device)

    def __call__(self, kptdescs):
        # setup data for superglue
        logging.debug("prepare input data for superglue...")
        data = {}
        data['image_size0'] = torch.from_numpy(kptdescs["ref"]["image_size"]).float().to(self.device)
        data['image_size1'] = torch.from_numpy(kptdescs["cur"]["image_size"]).float().to(self.device)

        if "torch" in kptdescs["cur"]:
            data['scores0'] = kptdescs["ref"]["torch"]["scores"][0].unsqueeze(0)
            data['keypoints0'] = kptdescs["ref"]["torch"]["keypoints"][0].unsqueeze(0)
            data['descriptors0'] = kptdescs["ref"]["torch"]["descriptors"][0].unsqueeze(0)

            data['scores1'] = kptdescs["cur"]["torch"]["scores"][0].unsqueeze(0)
            data['keypoints1'] = kptdescs["cur"]["torch"]["keypoints"][0].unsqueeze(0)
            data['descriptors1'] = kptdescs["cur"]["torch"]["descriptors"][0].unsqueeze(0)
        else:
            data['scores0'] = torch.from_numpy(kptdescs["ref"]["scores"]).float().to(self.device).unsqueeze(0)
            data['keypoints0'] = torch.from_numpy(kptdescs["ref"]["keypoints"]).float().to(self.device).unsqueeze(0)
            data['descriptors0'] = torch.from_numpy(kptdescs["ref"]["descriptors"]).float().to(self.device).unsqueeze(
                0).transpose(1, 2)

            data['scores1'] = torch.from_numpy(kptdescs["cur"]["scores"]).float().to(self.device).unsqueeze(0)
            data['keypoints1'] = torch.from_numpy(kptdescs["cur"]["keypoints"]).float().to(self.device).unsqueeze(0)
            data['descriptors1'] = torch.from_numpy(kptdescs["cur"]["descriptors"]).float().to(self.device).unsqueeze(
                0).transpose(1, 2)

        # Forward !!
        logging.debug("matching keypoints with superglue...")
        print("matching keypoints with superglue...")
        pred = self.superglue(data)

        # get matching keypoints
        kpts0 = kptdescs["ref"]["keypoints"]
        kpts1 = kptdescs["cur"]["keypoints"]

        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().detach().numpy()

        # Sort them in the order of their confidence.
        match_conf = []
        for i, (m, c) in enumerate(zip(matches, confidence)):
            match_conf.append([i, m, c])
        match_conf = sorted(match_conf, key=lambda x: x[2], reverse=True)

        valid = [[l[0], l[1]] for l in match_conf if l[1] > -1]
        v0 = [l[0] for l in valid]
        v1 = [l[1] for l in valid]
        mkpts0 = kpts0[v0]
        mkpts1 = kpts1[v1]

        ret_dict = {
            "ref_keypoints": mkpts0,
            "cur_keypoints": mkpts1,
            "match_score": confidence[v0]
        }

        return ret_dict

class MatcherNode:
    def __init__(self):
        rospy.init_node("superglue_matcher", anonymous=True)
        self.kptdescs = {}
        self.imgs = {}
        self.RATE = 60
        self.cnt_left = 0
        self.cnt_right = 0
        self.stereo = True
        self.use_timer = True
        self.mutex = Lock()

#       init rospy publishers
        self.ref_keypoints_pub = rospy.Publisher("/superglue/matches/ref_keypoints", String, queue_size=50)
        self.cur_keypoints_pub = rospy.Publisher("/superglue/matches/cur_keypoints", String, queue_size=50)
        self.match_score_pub = rospy.Publisher("/superglue/matches/match_score", String, queue_size=50)
        self.all_data_left_pub = rospy.Publisher("/superglue/matches/all_data_left", String, queue_size=50)  
        self.all_data_right_pub = rospy.Publisher("/superglue/matches/all_data_right", String, queue_size=50) 

#       init rospy subscribers
        rospy.Subscriber("/stereo/left/image_rect", Image, self.image_left_callback, queue_size = 1)
        if self.stereo:
            rospy.Subscriber("/stereo/right/image_rect", Image, self.image_right_callback, queue_size = 1)

        # rospy.sleep(3)
        if self.use_timer:
            self.timer = rospy.Timer(rospy.Duration(1. / self.RATE), self.timer_callback)
        else:
            self.timer = None

        if self.timer is None:
            self.image_left_buf = []
            self.image_right_buf = []
        else:
            self.image_left_buf = deque(maxlen=10000)
            self.image_right_buf = deque(maxlen=10000)


    def image_left_callback(self, msg):
        # if self.cnt_left%10 == 0:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        img = np.squeeze(img)

        # print("LEFT:", self.cnt_left)

        self.image_left_buf.append([msg.header.stamp, img])
        if self.timer is None:
            if len(self.image_left_buf) == 30:
                with self.mutex:
                    self.process_all(self.image_left_buf, cam_type="left")
        self.cnt_left += 1


    def image_right_callback(self, msg):
        # if self.cnt_right%10 == 0:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        img = np.squeeze(img)

        # print("RIGHT:", self.cnt_left)

        self.image_right_buf.append([msg.header.stamp, img])
        if self.timer is None:
            if len(self.image_right_buf) == 30:
                with self.mutex:
                    self.process_all(self.image_right_buf, cam_type="right")
        self.cnt_right += 1


    def timer_callback(self, event):
        with self.mutex:
            if self.image_left_buf:
                start_time = time.time()
                self.process(*(self.image_left_buf[0]), "left")
                # print('Time for left image:', time.time()-start_time)
                self.image_left_buf.popleft()

            if self.stereo:
                if self.image_right_buf:
                    start_time = time.time()
                    self.process(*(self.image_right_buf[0]), "right")
                    # print('Time for right image:', time.time()-start_time)
                    self.image_right_buf.popleft()
        

    
    def process_all(self, data, cam_type):
        print("Processing " + cam_type + "...")
        for img in tqdm(data):
            self.process(*img, cam_type)


    def process(self, t, img, cam_type):
        self.imgs["cur"] = img
        self.kptdescs["cur"] = detector(img)

        print("Frame " + cam_type + " time: ", t.to_sec())
        
        if "ref" in self.kptdescs:
            matches = matcher(self.kptdescs)
            img = plot_matches(self.imgs['ref'], self.imgs['cur'],
                                matches['ref_keypoints'][0:200], matches['cur_keypoints'][0:200],
                                matches['match_score'][0:200], layout='lr')
            #cv2.imshow("track", img)
            cv2.imwrite('matcher_' + cam_type + '.jpg', img)
            encoded_data_ref_keypoints = json.dumps(matches['ref_keypoints'].tolist())
            encoded_data_cur_keypoints = json.dumps(matches['cur_keypoints'].tolist())
            encoded_data_match_score = json.dumps(matches['match_score'].tolist())

            all_data = [matches['ref_keypoints'].tolist()] + [matches['cur_keypoints'].tolist()] + [matches['match_score'].tolist()] + [str(t)]
            encoded_data_all = json.dumps(all_data)

            self.ref_keypoints_pub.publish(encoded_data_ref_keypoints)
            self.cur_keypoints_pub.publish(encoded_data_cur_keypoints)
            self.match_score_pub.publish(encoded_data_match_score)

            if cam_type == "left":
                self.all_data_left_pub.publish(encoded_data_all)
            if cam_type == "right":
                self.all_data_right_pub.publish(encoded_data_all)
            
            # loaded_dictionary = json.loads(encoded_data_string)

        self.kptdescs["ref"], self.imgs["ref"] = self.kptdescs["cur"], self.imgs["cur"]


if __name__ == "__main__":
    detector = SuperPointDetector({"cuda": 1})
    matcher = SuperGlueMatcher({"cuda": 1, "weights": "outdoor"})

    matcher_node = MatcherNode()
    rospy.spin()

