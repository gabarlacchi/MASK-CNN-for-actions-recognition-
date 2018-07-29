import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import cv2

# import coco
import utils
import model as modellib
import visualize

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
# not interesting
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig():
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50
    BACKBONE = "resnet101"
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    NUM_CLASSES = 1 + 80  # Override in sub-classes
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_STRIDE = 1
    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported
    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    TRAIN_ROIS_PER_IMAGE = 200
    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33
    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100
    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7
    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    USE_RPN_ROIS = True

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        self.IMAGE_SHAPE = np.array(
            [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # Compute backbone size from input image size
        self.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
             for stride in self.BACKBONE_STRIDES])

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

def draw_shape(image, x1, y1, x2, y2, color):
    """Draws a shape from the given specs."""
    # Get the center x, y and the size s
    cv2.rectangle(image, (x1, y1), (x2, y2), 255, -1)
    image = image.reshape(240, 320)
    return image


config = InferenceConfig()
#config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# load all the labelled images
# activities = [dir for dir in os.listdir('ucf24_project/labels') if not dir.endswith('.DS_Store')]
activities = ['WalkingWithDog', 'VolleyballSpiking']
for act in activities:
    videos = [dir for dir in os.listdir('ucf24_project/labels/'+str(act)) if not dir.endswith('.DS_Store')]
    for vid in videos:
        frames = [dir.replace('.txt', '.jpg') for dir in os.listdir('ucf24_project/labels/' + str(act)+ '/' + str(vid)) if dir.endswith('.txt')]
        print( str(act) + ' ' + str(vid))
        for frame in frames:
            # get frame paht and set outupt path
            frame_dir = 'ucf24_project/rgb-images/'+str(act)+'/'+str(vid)+'/'+frame
            output_dir = 'ucf24_project/labels/'+str(act)+'/'+str(vid)+'/'

            # read the target frame
            image = skimage.io.imread(frame_dir)
            # run detectron on the image readed
            results = model.detect([image], verbose=1)
            # get the result
            r = results[0]
            # get the masks for the person classes
            indexes = np.where(r['class_ids'] == 1)[0]
            # collect the masks in the frame
            masks = []
            num_masks = 0
            for ind in indexes:
                masks.append(r['masks'][:,:,num_masks]*255)
                num_masks = num_masks + 1
            for k in range(num_masks):
                a_mask = masks[k]
                im = Image.fromarray(a_mask)
                im.save(output_dir+frame.replace('.jpg', '')+'mask_'+str(k)+".jpg")
