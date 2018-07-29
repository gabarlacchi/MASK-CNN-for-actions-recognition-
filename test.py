import matplotlib
matplotlib.use('TkAgg')


import os
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
from actionCLSS_dataset import ShapesDataset
import utils
import config
import model as modellib
import visualize
from model import log
from PIL import Image
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""
# Root directory of the project
ROOT_DIR = os.getcwd()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs/tr_heads_nativ_nativ_0.0005||2018-07-16")
# MODEL_DIR = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
config = actionCLSS_Config()
config.display()

dataset_test = ShapesDataset()
dataset_test.load_shapes(10, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], purpose='test')
dataset_test.prepare()

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

image_id = random.choice(dataset_test.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_test, inference_config,
                            image_id, use_mini_mask=False)

# USE HD IMAGES
# hd_img = plt.imread('HDimages_2/'+random.choice([img for img in os.listdir('HDimages_2/') if not img.endswith('.DS_Store')]))

hd_img = plt.imread('HDimages_2/skate.jpg')

hd_img , hd_window, hd_scale, hd_padding = utils.resize_image(
                                                hd_img,
                                                min_dim=config.IMAGE_MIN_DIM,
                                                max_dim=config.IMAGE_MAX_DIM,
                                                padding=config.IMAGE_PADDING)
# END HD IMAGES
results = model.detect([hd_img], verbose=1)


r = results[0]
N = r['rois'].shape[0]

for i in range(N):
    class_id = r['class_ids'][i]
    score = r['scores'][i] if r['scores'] is not None else None
    label = dataset_test.class_names[class_id]
    print(str(label) + " – " +str(score))


class_names = ['Background', 'WalkingWithDog', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving',
                'Fencing', 'FloorGymnastics', 'GolfSwing','HorseRiding', 'IceDancing', 'LongJump',
                'PoleVault', 'RopeClimbing', 'SalsaSpin','SkateBoarding', 'Skiing', 'Skijet',
                'SoccerJuggling', 'Surfing', 'TennisSwing','TrampolineJumping', 'VolleyballSpiking', 'Basketball']

visualize.display_instances(hd_img, r['rois'], r['masks'], r['class_ids'],
                              class_names, r['scores'], ax=get_ax())

