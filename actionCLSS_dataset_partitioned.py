import matplotlib

matplotlib.use('TkAgg')

import utils
import random
import glob, os
# import math
import cv2
import csv
# from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt


# import matplotlib.patches as patches
# import matplotlib.lines as lines
# from matplotlib.patches import Polygon
# import IPython.display


class ShapesDatasetPartitioned(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    with open('validationList.txt') as file:
        reserved_for_val = [video.replace('\n', '') for video in file.readlines()]
    with open('testList.txt') as file:
        reserved_for_test = [video.replace('\n', '') for video in file.readlines()]

    def load_train_shapes(self):
        # Training is True if the dataset is for training, False if it's for validation set
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("sport", 1, "WalkingWithDog")
        self.add_class("sport", 2, "BasketballDunk")
        self.add_class("sport", 3, "Biking")
        self.add_class("sport", 4, "CliffDiving")
        self.add_class("sport", 5, "CricketBowling")
        self.add_class("sport", 6, "Diving")
        self.add_class("sport", 7, "Fencing")
        self.add_class("sport", 8, "FloorGymnastics")
        self.add_class("sport", 9, "GolfSwing")
        self.add_class("sport", 10, "HorseRiding")
        self.add_class("sport", 11, "IceDancing")
        self.add_class("sport", 12, "LongJump")
        self.add_class("sport", 13, "PoleVault")
        self.add_class("sport", 14, "RopeClimbing")
        self.add_class("sport", 15, "SalsaSpin")
        self.add_class("sport", 16, "SkateBoarding")
        self.add_class("sport", 17, "Skiing")
        self.add_class("sport", 18, "Skijet")
        self.add_class("sport", 19, "SoccerJuggling")
        self.add_class("sport", 20, "Surfing")
        self.add_class("sport", 21, "TennisSwing")
        self.add_class("sport", 22, "TrampolineJumping")
        self.add_class("sport", 23, "VolleyballSpiking")
        self.add_class("sport", 24, "Basketball")

        # Select ALL IMAGES
        root_labels = 'ucf24_project/labels'
        root_images = 'ucf24_project/rgb-images'
        for activity in [act['name'] for act in self.class_info if not act['name'] == 'BackGround']:
            for video in [vid for vid in os.listdir(root_labels + '/' + activity) if
                          not vid.endswith('.DS_Store')
                          and vid not in self.reserved_for_test
                          and vid not in self.reserved_for_val]:
                for frame in [f for f in os.listdir(root_labels + '/' + activity + '/' + video) if
                              f.endswith('.txt')]:

                    URL_label = root_labels + '/' + activity + '/' + video + '/' + frame
                    URL_image = root_images + '/' + activity + '/' + video + '/' + frame.replace('.txt', '.jpg')
                    # open the file with the labels (class and bbox)
                    with open(URL_label) as file:
                        lines = file.readlines()
                        bounding_boxes = [str(line).replace('\n', '').split(' ') for line in lines]
                        # Orrible cast from string matrix to int matrix
                        for i in range(len(bounding_boxes)):
                            bounding_boxes[i] = [int(float(element)) for element in bounding_boxes[i]]

                    # add image
                    self.add_image("sport", image_id=i, path=URL_image,
                                   width=320, height=240,
                                   bbox=bounding_boxes, action=activity)


    def load_val_shapes(self):
        # Training is True if the dataset is for training, False if it's for validation set
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("sport", 1, "WalkingWithDog")
        self.add_class("sport", 2, "BasketballDunk")
        self.add_class("sport", 3, "Biking")
        self.add_class("sport", 4, "CliffDiving")
        self.add_class("sport", 5, "CricketBowling")
        self.add_class("sport", 6, "Diving")
        self.add_class("sport", 7, "Fencing")
        self.add_class("sport", 8, "FloorGymnastics")
        self.add_class("sport", 9, "GolfSwing")
        self.add_class("sport", 10, "HorseRiding")
        self.add_class("sport", 11, "IceDancing")
        self.add_class("sport", 12, "LongJump")
        self.add_class("sport", 13, "PoleVault")
        self.add_class("sport", 14, "RopeClimbing")
        self.add_class("sport", 15, "SalsaSpin")
        self.add_class("sport", 16, "SkateBoarding")
        self.add_class("sport", 17, "Skiing")
        self.add_class("sport", 18, "Skijet")
        self.add_class("sport", 19, "SoccerJuggling")
        self.add_class("sport", 20, "Surfing")
        self.add_class("sport", 21, "TennisSwing")
        self.add_class("sport", 22, "TrampolineJumping")
        self.add_class("sport", 23, "VolleyballSpiking")
        self.add_class("sport", 24, "Basketball")

        # Select ALL IMAGES
        root_labels = 'ucf24_project/labels'
        root_images = 'ucf24_project/rgb-images'
        for activity in [act['name'] for act in self.class_info if not act['name'] == 'BackGround']:
            for video in [vid for vid in os.listdir(root_labels + '/' + activity) if
                          not vid.endswith('.DS_Store')
                          and vid in self.reserved_for_val]:
                for frame in [f for f in os.listdir(root_labels + '/' + activity + '/' + video) if
                              f.endswith('.txt')]:

                    URL_label = root_labels + '/' + activity + '/' + video + '/' + frame
                    URL_image = root_images + '/' + activity + '/' + video + '/' + frame.replace('.txt', '.jpg')
                    # open the file with the labels (class and bbox)
                    with open(URL_label) as file:
                        lines = file.readlines()
                        bounding_boxes = [str(line).replace('\n', '').split(' ') for line in lines]
                        # Orrible cast from string matrix to int matrix
                        for i in range(len(bounding_boxes)):
                            bounding_boxes[i] = [int(float(element)) for element in bounding_boxes[i]]

                    # add image
                    self.add_image("sport", image_id=i, path=URL_image,
                                   width=320, height=240,
                                   bbox=bounding_boxes, action=activity)



    def load_test_shapes(self):
        # Training is True if the dataset is for training, False if it's for validation set
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("sport", 1, "WalkingWithDog")
        self.add_class("sport", 2, "BasketballDunk")
        self.add_class("sport", 3, "Biking")
        self.add_class("sport", 4, "CliffDiving")
        self.add_class("sport", 5, "CricketBowling")
        self.add_class("sport", 6, "Diving")
        self.add_class("sport", 7, "Fencing")
        self.add_class("sport", 8, "FloorGymnastics")
        self.add_class("sport", 9, "GolfSwing")
        self.add_class("sport", 10, "HorseRiding")
        self.add_class("sport", 11, "IceDancing")
        self.add_class("sport", 12, "LongJump")
        self.add_class("sport", 13, "PoleVault")
        self.add_class("sport", 14, "RopeClimbing")
        self.add_class("sport", 15, "SalsaSpin")
        self.add_class("sport", 16, "SkateBoarding")
        self.add_class("sport", 17, "Skiing")
        self.add_class("sport", 18, "Skijet")
        self.add_class("sport", 19, "SoccerJuggling")
        self.add_class("sport", 20, "Surfing")
        self.add_class("sport", 21, "TennisSwing")
        self.add_class("sport", 22, "TrampolineJumping")
        self.add_class("sport", 23, "VolleyballSpiking")
        self.add_class("sport", 24, "Basketball")

        # Select ALL IMAGES
        root_labels = 'ucf24_project/labels'
        root_images = 'ucf24_project/rgb-images'
        for activity in [act['name'] for act in self.class_info if not act['name'] == 'BackGround']:
            for video in [vid for vid in os.listdir(root_labels + '/' + activity) if
                          not vid.endswith('.DS_Store')
                          and vid in self.reserved_for_test]:
                for frame in [f for f in os.listdir(root_labels + '/' + activity + '/' + video) if
                              f.endswith('.txt')]:

                    URL_label = root_labels + '/' + activity + '/' + video + '/' + frame
                    URL_image = root_images + '/' + activity + '/' + video + '/' + frame.replace('.txt', '.jpg')
                    # open the file with the labels (class and bbox)
                    with open(URL_label) as file:
                        lines = file.readlines()
                        bounding_boxes = [str(line).replace('\n', '').split(' ') for line in lines]
                        # Orrible cast from string matrix to int matrix
                        for i in range(len(bounding_boxes)):
                            bounding_boxes[i] = [int(float(element)) for element in bounding_boxes[i]]

                    # add image
                    self.add_image("sport", image_id=i, path=URL_image,
                                   width=320, height=240,
                                   bbox=bounding_boxes, action=activity)


    def load_image(self, image_id):

        info = self.image_info[image_id]
        image = plt.imread(info['path'])

        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['bbox']
        masks_path = info['path'].replace(".jpg", "mask_").replace("rgb-images", "labels")
        count = len(shapes)
        # si crea una matrice 3D per ogni bbox. La terza dimensione è il numero di bounding.
        # ogni maschera è un area di 0 su un blocco di 1 grosso come l'immagine (320x240)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)

        for i, (action, x1, y1, x2, y2) in enumerate(info['bbox']):
            mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(), x1, y1, x2, y2, 1).reshape(240, 320, 1)
            # se ci sono le maschere...
            if os.path.exists(masks_path + "0.jpg"):
                mask_counter, intersection_max = 0, -1
                # per ogni maschera
                while os.path.exists(masks_path + str(mask_counter) + ".jpg"):
                    mask_counter = mask_counter + 1
                    # leggo la maschera, essendo in jpg è compressa, ci sta che non tutti i valori siano 255
                    # uso ">128" per filtrarla -> ottengo una matrice binaria
                    current_mask = plt.imread(masks_path + "0.jpg") > 128
                    # Calcolo l'intersezione col boundingbox e la percentuale di maschera dentro questo
                    intersection = mask[:, :, i:i + 1] * current_mask.reshape(240, 320, 1)
                    # the factors *1 and /255 needed in order to had a normalized score
                    intersection_score = sum(sum(intersection / 255)) / sum(sum(current_mask * 1))
                    # Se ho un nuovo massimo, metto da parte l'intersezione trovata
                    if intersection_score > intersection_max:
                        intersection_max = intersection_score
                        final_mask = current_mask
                # Se ho avuto almeno un intersezione con il bbox
                if intersection_max > 0.1:
                    # uncomment to save a mask, but seems good
                    # print(intersection_max)
                    aMask = mask[:, :, i:i + 1] * final_mask.reshape(240, 320, 1)
                    # plt.imsave("imgSaved/"+str(intersection_max)+"m.png", aMask.reshape(240, 320))
                    # plt.imsave("imgSaved/"+str(intersection_max)+"i.png", plt.imread(info['path']))
                    # plt.imsave("imgSaved/"+str(intersection_max)+"b.png", mask[:, :, i:i+1].reshape(240, 320))
                    # aggiorno la maschera
                    mask[:, :, i:i + 1] = mask[:, :, i:i + 1] * final_mask.reshape(240, 320, 1)

        # prima poteva accadere che due maschere in scena fossero di classi diverse (e.g. tondo e quadrato)
        # ora avremo che le maschere in scena fanno tutte parte della stessa attività
        class_ids = np.array([self.class_names.index(info['action']) for i in range(count)])

        return mask, class_ids.astype(np.int32)

    def draw_shape(self, image, x1, y1, x2, y2, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        cv2.rectangle(image, (x1, y1), (x2, y2), 255, -1)
        image = image.reshape(240, 320)

        return image
