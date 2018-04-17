from config import Config

class actionCLSS_Config(Config):
	""" Configuration for training on the action oriented images dataset.
		Derives from the base config class and overrides values specific to our own dataset.
	"""

	NAME = "actions"

	# Train on 1 GPU and 8 images. We can put few images in a single GPU only if the images are small
	# At the start use 1 GPU and small images for a faster training. Set values in different way later
	GPU_COUNT = 1
	IMAGEG_PER_GPU = 8

	NUM_CLASSES = 24

	# Use small images for faster training. Set values in different way later
	IMAGE_MIN_DIM = 128
	IMAGE_MAX_DIM = 128

	# The anchors for training bounding boxes
	RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

	# Reduce ROIS 'cause the smaller images and we have only few objects in the images
	TRAIN_ROIS_PER_IMAGE = 32

	# for now we set few steps for a faster training
	STEPS_PER_EPOCH = 150

	# the same as above
	VALIDATION_STEPS = 5




