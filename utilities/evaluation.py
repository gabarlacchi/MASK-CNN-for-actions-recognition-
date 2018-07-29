import matplotlib
matplotlib.use('TkAgg')

import os
import pickle
import sys
import random
import math
import re
import time
import numpy as np
import cv2
#import matplotlib
import matplotlib.pyplot as plt

from actionCLSS_config import actionCLSS_Config
from actionCLSS_dataset_partitioned import ShapesDatasetPartitioned
import utils
import config
import model as modellib
import visualize
from model import log
from PIL import Image
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Root directory of the project
ROOT_DIR = os.getcwd()

MODEL_DIR = os.path.join(ROOT_DIR, "logs/tr_heads_nativ_nativ_0.0005||2018-07-16")
config = actionCLSS_Config()

dataset_test = ShapesDatasetPartitioned()
# Quanto è grande il test set?
dataset_test.load_test_shapes()
dataset_test.prepare()

class_names = ['WalkingWithDog', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving',
                'Fencing', 'FloorGymnastics', 'GolfSwing','HorseRiding', 'IceDancing', 'LongJump',
                'PoleVault', 'RopeClimbing', 'SalsaSpin','SkateBoarding', 'Skiing', 'Skijet',
                'SoccerJuggling', 'Surfing', 'TennisSwing','TrampolineJumping', 'VolleyballSpiking', 'Basketball', 'Background']

class InferenceConfig(actionCLSS_Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                           config=inference_config,
                           model_dir=MODEL_DIR)

model_path = model.find_last()[1]
model.load_weights(model_path, by_name=True)


# Compute VOC-Style mAP @ IoU=0.5
# compute the entire dataset
image_ids = dataset_test.image_ids
APs = []

ev_ids = []
ev_scores = []
ev_pred_match = []
ev_gt_match = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_test, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]

    #IF PREDICT SOMETHING
    if not r['masks'].shape[0] == 0:

        # Compute AP
        AP, precisions, recalls, overlaps, pred_match, gt_match = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r["masks"], iou_threshold=0.25)

        ev_ids.append(r['class_ids'])
        ev_scores.append(r['scores'])
        ev_pred_match.append(pred_match)
        ev_gt_match.append(gt_match)
        # print((gt_match))
        APs.append(AP)

    if image_id%500 == 1 or image_id == len(image_ids):
        print(image_id)
        with open('evaluationVarsM/class_ids.pkl', 'wb') as f:
            pickle.dump(ev_ids, f)

        with open('evaluationVarsM/scores.pkl', 'wb') as f:
            pickle.dump(ev_scores, f)

        with open('evaluationVarsM/pred_matches.pkl', 'wb') as f:
            pickle.dump(ev_pred_match, f)

        with open('evaluationVarsM/gt_matches.pkl', 'wb') as f:
            pickle.dump(ev_gt_match, f)



print("mAP @ IoU=50: ", np.mean(APs))

