#!/usr/bin/env python3

import os.path as path
import logging
import cv2
from tools.tools import *
from superpoint import SuperPoint


class SuperPointDetector(object):
    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "path": path.abspath(path.join(__file__ ,"../..")) + "/models/SuperPointPretrainedNetwork/superpoint_v1.pth",
        "cuda": True
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("SuperPoint detector config: ")
        logging.info(self.config)

        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'

        logging.info("creating SuperPoint detector...")
        self.superpoint = SuperPoint(self.config).to(self.device)

    def __call__(self, image):
        try:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except: pass # Squeezed gray image array

        logging.debug("detecting keypoints with superpoint...")
        image_tensor = image2tensor(image, self.device)
        # print(image_tensor.shape)
        pred = self.superpoint({'image': image_tensor})

        ret_dict = {
            "image_size": np.array([image.shape[0], image.shape[1]]),
            "torch": pred,
            "keypoints": pred["keypoints"][0].cpu().detach().numpy(),
            "scores": pred["scores"][0].cpu().detach().numpy(),
            "descriptors": pred["descriptors"][0].cpu().detach().numpy().transpose()
        }

        #print(ret_dict)

        return ret_dict


if __name__ == "__main__":
    img = cv2.imread( path.abspath(path.join(__file__ ,"../../../..")) +"/test.jpg")

    detector = SuperPointDetector()
    kptdescs = detector(img)

    img = plot_keypoints(img, kptdescs["keypoints"], kptdescs["scores"])
    cv2.imshow("SuperPoint", img)
    cv2.waitKey()