#!/usr/bin/env python3

import os.path as path
from tools.tools import *
import logging
from models.superglue import SuperGlue
import rospy
import numpy as np
from sensor_msgs.msg import Image
import time


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

prev_time = None
time_list = []
kptdescs = {}
imgs = {}


def process_image(msg):
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

    img = np.squeeze(img)

    global prev_time
    global time_list

    global kptdescs
    global imgs
    
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

    imgs["cur"] = img
    kptdescs["cur"] = detector(img)
    
    if "ref" in kptdescs:
        matches = matcher(kptdescs)
        img = plot_matches(imgs['ref'], imgs['cur'],
                            matches['ref_keypoints'][0:200], matches['cur_keypoints'][0:200],
                            matches['match_score'][0:200], layout='lr')
        #cv2.imshow("track", img)
        cv2.imwrite('matcher.jpg', img)

    kptdescs["ref"], imgs["ref"] = kptdescs["cur"], imgs["cur"]

    
    

if __name__ == "__main__":
    from detector import SuperPointDetector

    detector = SuperPointDetector({"cuda": 0})
    matcher = SuperGlueMatcher({"cuda": 0, "weights": "outdoor"})

    rospy.init_node('superglue_matcher', anonymous=True)

    #rospy.Subscriber("/stereo/left/image_raw", Image, process_image)
    rospy.Subscriber("/stereo/left/image_rect", Image, process_image, queue_size = 1)
    
    rospy.spin()

