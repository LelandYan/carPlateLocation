import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import skimage.io as io
import cv2

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN"))  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import carplate

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
CARPLATE_WEIGHTS_PATH = "mask_rcnn_carplate_0030.h5"

# 车牌定位

# %%

config = carplate.CarplateConfig()
ROOT__DATA_DIR = os.path.abspath("../")
CARPLATE_DIR = os.path.join(ROOT__DATA_DIR, r"dataset\carplate")


# %%

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
TEST_MODE = "inference"


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# Load validation dataset
dataset = carplate.CarplateDataset()
dataset.load_carplate(CARPLATE_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
# Load Mask-RCNN Model
# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights
weights_path = CARPLATE_WEIGHTS_PATH
print("Loading weights", weights_path)
model.load_weights(weights_path, by_name=True)
# Run Detection
image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       dataset.image_reference(image_id)))

visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, dataset.class_names)
results = model.detect([image], verbose=1)
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
# 1.提取车牌区域，只取第一个车牌
mask = r['masks'][:, :, 0].astype(np.uint8)
plt.show()
_, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
epsilon = 0.1 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
approx = approx.squeeze()

if approx.shape == (4, 2):
    box = np.zeros_like(approx)
    box[:, 0] = approx[:, 1]
    box[:, 1] = approx[:, 0]
else:
    rect = cv2.minAreaRect(np.array(np.nonzero(mask)).T)
    box = cv2.boxPoints(rect).astype(np.int)

y0, x0 = box.min(axis=0)
y1, x1 = box.max(axis=0)
img = image[y0:y1, x0:x1]
io.imshow(img)
plt.show()
box[:, 0] -= y0
box[:, 1] -= x0
# 调整box顺序，从左上角开始，逆时针转动
i0 = (box[:, 0] + box[:, 1]).argmin()
box = box[[i0, (i0 + 1) % 4, (i0 + 2) % 4, (i0 + 3) % 4]]
plt.scatter(box[:, 1], box[:, 0], c='r')
# 2.矫正车牌
h = np.max([box[1][0] - box[0][0], box[2][0] - box[3][0]])
w = np.max([box[2][1] - box[1][1], box[3][1] - box[0][1]])
box2 = np.array([(0, 0), (h, 0), (h, w), (0, w)])
M = cv2.getPerspectiveTransform(box[:, ::-1].astype(np.float32), box2[:, ::-1].astype(np.float32))
img = cv2.warpPerspective(img, M, (w, h))
img = cv2.resize(img, (220, 70))
io.imshow(img)
plt.show()
