import matplotlib
matplotlib.use('TkAgg')

import utils
import random
import glob, os
import cv2
import csv
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import IPython.display

class actionCLSS_Dataset(utils.Dataset):
	""" Import images and create the training dataset. Images are in the path
		ucf24/rgb-images
		The total classe are 24
	"""

	paths = []
	actions = []
	boxes = []
	classes = [
			"Basketball",
			"BasketballDunk",
			"Biking",
			 "CliffDiving",
			 "CricketBowling",
			 "Diving",
			 "Fencing",
			 "FloorGymnastics",
			 "GolfSwing",
			 "HorseRiding",
			 "IceDancing",
			 "LongJump",
			 "PoleVault",
			 "RopeClimbing",
			 "SalsaSpin",
			 "SkateBoarding",
			 "Skiing",
			 "Skijet",
			 "SoccerJuggling",
			 "Surfing",
			 "TennisSwing",
			 "TrampolineJumping",
			 "VolleyballSpiking",
			 "WalkingWithDog"
		]

	def load_dataset(self, count, width, height):
		# add the new classes
		for i in range( len(self.classes) ):
			self.add_class("actions", i, self.classes[i])

		# getting images and add them to the Dataset
		for i in range(count):
			action = self.selectActions()
			# self.add_image("actions", image_id=i, path=self.paths[i], width=width, height=height, actions=self.actions)
			#print(actions)
			#print(self.boxes[i])
			#print('iter: ' + str(i))
			self.add_image("actions", image_id=i, path=self.paths[i], width=width, height=height, actions=action, bbox=self.boxes[i])


	def selectActions(self):
		# Select random path of the folders
		root = 'ucf24_project/labels/'
		activity = random.choice([dir for dir in os.listdir(root) if not dir.endswith('.DS_Store')])
		video = random.choice([dir for dir in os.listdir(root+'/'+activity) if not dir.endswith('.DS_Store')])
		frame = random.choice([f for f in os.listdir(root+'/'+activity+'/'+video) if not f.endswith('.DS_Store')])
		URL = root+'/'+activity+'/'+video+'/'+frame
		# open the file with the labels (class and bbox)
		with open(URL) as file:
			lines = file.readlines()
			_tuple = [str(line).replace('\n', '').split(' ') for line in lines]
		# bb
		self.boxes.append(_tuple[0][1:5])
		# self.actions.append((self.classes[indexClass], 'color', dims))
		self.actions.append(activity)
		self.paths.append(URL)
		dims = 1, 2, 3
		# not necessary (just not to change the code)
		color = tuple([random.randint(0, 255) for _ in range(3)])
		
		return activity




	def return_bbox(self, image_id):
		b = np.zeros([1, 4], dtype=np.int32)
		boxes = [float(x) for x in self.boxes[image_id]]
		# np.array([y1, x1, y2, x2])
		b[0] = np.array([boxes[1], boxes[0], boxes[3], boxes[2]])
		return b.astype(np.int32)
	
	# override method
	def load_image(self, image_id):
		# retrieve image from folder
		# the url of the file, different from the path of the image
		# the path is like ucf24_project/labels/LongJump/v_LongJump_g19_c01/00066.txt
		file_url = self.paths[image_id]
		img_url = (file_url.replace('labels', 'rgb-images')).replace('txt', 'jpg')
		image = np.asarray(Image.open(img_url), dtype=np.uint8)
		return image
		
	# override method
	def load_mask(self, image_id):
		"""Generate instance masks for shapes of the given image ID.
		"""
		'''
		info = self.image_info[image_id]
		shapes = info['actions']
		count = len(shapes)
		mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
		for i, (shape, _, dims) in enumerate(info['actions']):
			mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)
		# Handle occlusions
		occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
		for i in range(count-2, -1, -1):
			mask[:, :, i] = mask[:, :, i] * occlusion
			occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
			# Map class names to class IDs.
		class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        
		return mask, class_ids.astype(np.int32)
		'''
		info = self.image_info[image_id]
		mask = np.zeros([info['height'], info['width'], 1], dtype=np.uint8)
		return mask, np.array([1])


	def draw_shape(self, image, shape, dims, color):
		"""Draws a shape from the given specs."""
		# Get the center x, y and the size s
		x, y, s = dims
		cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
		return image
	
	# override method
	def image_reference(self, image_id):
		"""Return the shapes data of the image."""
		info = self.image_info[image_id]
		if info["source"] == "actions":
			return info["actions"]
		else:
			super(self.__class__).image_reference(self, image_id)

	'''
	def load_bb(self, image, image_id):
		# draw the bounding box in the image
		# the input image is in pillow format
		draw = ImageDraw.Draw(image)
		draw.rectangle((float(self.boxes[image_id][0]), float(self.boxes[image_id][1]), 
						(float(self.boxes[image_id][2]), float(self.boxes[image_id][3]))))
		return image
	'''

	def draw_box(self, image, box, class_id):
		#x1, x2, x3, x4 = box
		#print(box)	
		_, ax = plt.subplots(1, figsize=(16,16))
		# Show area outside image boundaries.
		height, width = image.shape[:2]
		ax.set_ylim(height + 10, -10)
		ax.set_xlim(-10, width + 10)
		ax.axis('off')
		title = "RISULTATO"
		ax.set_title(title)
		y1, x1, y2, x2 = box[0][0], box[0][1], box[0][2], box[0][3]
		'''
		p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3,
								alpha=0.4, linestyle="dashed",
								facecolor='black')
		ax.add_patch(p)
		
		'''
		# Create a Rectangle patch
		rect = patches.Rectangle((x1,y1),x2 - x1,y2 - y1,linewidth=1,edgecolor='r',facecolor='none')

		# Add the patch to the Axes
		ax.add_patch(rect)
	
		#print (str(class_id))
		label = self.classes[int(class_id)]
		ax.text(x1, y1 + 8, label,
                color='w', size=11, backgroundcolor="none")
		ax.imshow(image.astype(np.uint8))
		plt.show()

		




	














