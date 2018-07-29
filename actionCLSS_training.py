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
import matplotlib.pyplot as plt
import utils
import config
from PIL import Image
from datetime import date

# from the project
from actionCLSS_config import actionCLSS_Config
from actionCLSS_dataset import ShapesDataset
from actionCLSS_dataset_partitioned import ShapesDatasetPartitioned
import model as modellib
import visualize
from model import log

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Root directory of the project
ROOT_DIR = os.getcwd()

# ! SAVING MODELS ! #
#   passing data to the function after loaded the images
#   If fold_name is empty ('') a default one are given. See below

def save_model(LAYERS, N_TR_IMGS, N_VAL_IMGS, learninRate, fold_name='', dir='logs/'):
    path = []
    # compose the name with defaults values. Add info or not below
    path.append(dir)
    path.append('tr_')
    path.append(str(LAYERS))
    path.append('_')
    path.append(str(N_TR_IMGS))
    path.append('_')
    path.append(str(N_VAL_IMGS))
    path.append('_')
    path.append(str(learninRate))
    path.append('||')
    if fold_name == '':
        path.append(str(date.today()))
    else:
        path.append(fold_name)
    full_path = ''.join(path)

    try:
        os.makedirs(full_path)
    except OSError:
        if not os.path.isdir(full_path):
            raise


    return full_path


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

config = actionCLSS_Config()
config.display()

TARGET_LAYERS = 'heads'


# training dataset
N_TRAIN_IMGS = 'nativ'
dataset_train = ShapesDatasetPartitioned()
dataset_train.load_train_shapes()
dataset_train.prepare()

# Validation dataset
N_VAL_IMGS = 'nativ'
dataset_val = ShapesDatasetPartitioned()
dataset_val.load_val_shapes()
dataset_val.prepare()

# Create directory to save logs and trained model
full_path = save_model(TARGET_LAYERS, N_TRAIN_IMGS, N_VAL_IMGS, config.LEARNING_RATE, '')
MODEL_DIR = os.path.join(ROOT_DIR, full_path)
print('MODEL SAVED PATH:' + str(full_path))

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"
#init_with = "last"
if init_with == "coco":
	model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                 "mrcnn_bbox", "mrcnn_mask"])


elif init_with == "last":
	# Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=20,
            layers=TARGET_LAYERS)
