from config import Config

class actionCLSS_Config(Config):
	""" Configuration for training on the action oriented images dataset.
		Derives from the base config class and overrides values specific to our own dataset.
	"""

	NAME = "SPORTS"

	# Train on 1 GPU and 8 images. We can put few images in a single GPU only if the images are small
	# At the start use 1 GPU and small images for a faster training. Set values in different way later
	GPU_COUNT = 1
	IMAGEG_PER_GPU = 8

	DETECTION_MIN_CONFIDENCE = 0.7
	LEARNING_RATE = 0.0005

	NUM_CLASSES = 1+24

	# Use small images for faster training. Set values in different way later
	IMAGE_MIN_DIM = 240
	IMAGE_MAX_DIM = 320

	# The anchors for training bounding boxes
	RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

	# Reduce ROIS 'cause the smaller images and we have only few objects in the images
	TRAIN_ROIS_PER_IMAGE = 32

	# for now we set few steps for a faster training
	STEPS_PER_EPOCH = 10000

	# the same as above
	VALIDATION_STEPS = 8000




