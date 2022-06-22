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
import skimage.draw
import glob
from skimage import img_as_uint, io

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import plot_precision_recall
import mrcnn.model as modellib
from mrcnn.model import log

from samples.Sclera import sclera
import glob
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import io, exposure, img_as_uint, img_as_float
import cv2

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
BALLON_WEIGHTS_PATH = "/home/user1/Desktop/kiran-sandbox/Mask_RCNN/samples/mask_rcnn_balloon.h5" #sys.argv[2] #"/home/user1/Desktop/kiran-sandbox/Mask_RCNN/logs/sclera20200810T1300/mask_rcnn_sclera_0100.h5"  # TODO: update this path
config = sclera.scleraConfig()
BALLOON_DIR = "/home/user1/Desktop/kiran-sandbox/Mask_RCNN/samples/Sclera/dataset/"#os.path.join(ROOT_DIR, "dataset")

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

infer_config = InferenceConfig()
infer_config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
    
# Load validation dataset
#dataset = sclera.scleraDataset()
#dataset.load_sclera(BALLOON_DIR, "val")

# Must call before using the dataset
#dataset.prepare()

#print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
'''
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
'''
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=infer_config)
                              
# Set path to balloon weights file

# Download file from the Releases page and set its path
# https://github.com/matterport/Mask_RCNN/releases
# weights_path = "/path/to/mask_rcnn_balloon.h5"

# Or, load the last model you trained
weights_path = "/home/user1/Desktop/kiran-sandbox/Mask_RCNN/samples/Sclera/weights/SBVPI/resnet_aug_resize/mask_rcnn_sclera_0080.h5" #model.find_last()

result_dir = sys.argv[1]
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

mask_result_dir = result_dir + "/Binarised/"
if not os.path.exists(mask_result_dir):
    os.mkdir(mask_result_dir)

prediction_result_dir = result_dir + "/Predictions/"
if not os.path.exists(prediction_result_dir):
    os.mkdir(prediction_result_dir)



logfile = open(BALLON_WEIGHTS_PATH.split('/')[-1] + "txt", 'w')

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
print("model loaded")
#model.keras_model.summary()
newsize = (480, 320)

image_id = 0
for img in glob.glob("/home/user1/Desktop/kiran-sandbox/Mask_RCNN/samples/Sclera/dataset/SSBC_FINAL/test-dataset/MOBIUS/orginal/*.jpg"):
	#print(img.split('/')[-1])
	#image_id = random.choice(dataset.image_ids)
	#image, image_meta, gt_class_id, gt_bbox, gt_mask =\
	#    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
	#info = dataset.image_info[image_id]
	#print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, img))
	image = skimage.io.imread(img)
	#skimage.io.imshow(image)
	#io.show()
	#source_file_name = "mask-input/source-" + str(image_id) + ".jpg"
	#source_file_name = result_dir + "/source-" + img.split('/')[-1]
	#skimage.io.imsave(source_file_name, image)
	# Run object detection
	results = model.detect([image], verbose=1)[0]
	get_name = img.split('/')[-1]#.replace('.jpg','')
	get_name = get_name.replace(".jpg", ".png")
	print(get_name)
	mask_file_name = mask_result_dir + get_name#img.split('/')[-1] #str(image_id) + ".jpg"
	print(mask_file_name)
	splash_file_name = prediction_result_dir + get_name#img.split('/')[-1]# str(image_id) + ".jpg"
	print(splash_file_name)
	mask_image = results['masks']
	mask_int = mask_image.astype(np.uint8)
	#mask_image = mask_image.resize(newsize)
	if mask_image.size != 0:
		try:
			print(mask_image.shape)
			image_get = skimage.color.gray2rgb(skimage.color.rgb2gray(image))
			splash = sclera.color_splash(image, results['masks'])
			splash = np.where(mask_int ,splash, image_get)

			skimage.io.imsave(splash_file_name, splash)
			print('splash dimesnion', splash.shape)
			#print('norm_img dimension', norm_img.shape)
			result = cv2.imread("/home/user1/Desktop/kiran-sandbox/Mask_RCNN/samples/Sclera/dataset/predict_test/5.png",0)
			x, y, w, h = 300, 300, 300,300
			ROI = result[y:y+h, x:x+w]
			print(ROI)


			# Calculate mean and STD
			mean, STD  = cv2.meanStdDev(ROI*4)
			image1 = cv2.imread(splash_file_name,0)
			offset = 0.8
			clipped = np.clip(image1, mean - offset*STD, mean + offset*STD).astype(np.uint8)
			get_normalize = cv2.normalize(clipped, clipped, 0, 254, norm_type=cv2.NORM_MINMAX)
			skimage.io.imsave(splash_file_name, get_normalize)
			skimage.io.imsave(mask_file_name, img_as_uint(mask_image))

		except:
			print("Failed val image :: " + img)
			logfile.write("%s\n" % img)
			new_img = Image.new('RGB',(480,320))
			new_img.save(mask_file_name, "PNG", optimize=True)
			image_get = skimage.color.gray2rgb(skimage.color.rgb2gray(image))*0
			skimage.io.imsave(splash_file_name, image_get)
			continue
	else:
		print("Failed val image :: " + img)
		logfile.write("%s\n" % img)
		new_img = Image.new('RGB',(480,320))
		new_img.save(mask_file_name, "PNG", optimize=True)#jpeg
		image_get = skimage.color.gray2rgb(skimage.color.rgb2gray(image))*0
		skimage.io.imsave(splash_file_name, image_get)
	image_id =  image_id + 1

#visualize.plot_precision_recall()

# result_dir = sys.argv[1] + "-contrast-enhanced"
# os.mkdir(result_dir)
# #model.keras_model.summary()
# image_id = 0
# for img in glob.glob("/home/user1/Desktop/kiran-sandbox/Mask_RCNN/samples/Sclera/dataset/test/ssbc-2020-challenge-dataset-test/SSBC-2020-MOBIUS-resized-contrast-adjusted/*.jpg"):
# 	print(img)
# 	#print(img.split('/')[-1])
# 	#image_id = random.choice(dataset.image_ids)
# 	#image, image_meta, gt_class_id, gt_bbox, gt_mask =\
# 	#    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
# 	#info = dataset.image_info[image_id]
# 	#print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, img))
# 	image = skimage.io.imread(img)
# 	#source_file_name = "mask-input/source-" + str(image_id) + ".jpg"
# 	source_file_name = result_dir + "/source-" + img.split('/')[-1]
# 	skimage.io.imsave(source_file_name, image)
# 	# Run object detection
# 	results = model.detect([image], verbose=1)[0]
# 	mask_image = results['masks']
# 	if mask_image.size != 0:
# 		try:
# 			print(mask_image.shape)
# 			splash = sclera.color_splash(image, results['masks'])
# 			# Save output
# 			splash_file_name = result_dir + "/splash-" + img.split('/')[-1]# str(image_id) + ".jpg"
# 			skimage.io.imsave(splash_file_name, splash)


# 			mask_file_name = result_dir + "/mask-" + img.split('/')[-1] #str(image_id) + ".jpg"
# 			skimage.io.imsave(mask_file_name, img_as_uint(mask_image))
				
# 			#confidence_score_image = results['scores']
# 			#print(confidence_score_image)
# 			#confidence_score_file_name = "resnet50-mask-output-val-set/confidence-" + img.split('/')[-1]# str(image_id) + ".jpg"
# 			#skimage.io.imsave(confidence_score_file_name, img_as_ubyte(confidence_score_image))
# 			# Display results
# 			#ax = get_ax(1)
# 			#r = results
# 			#visualize.display_instances(image, results['rois'], results['masks'], results['class_ids'], 
# 			#	                    dataset.class_names, results['scores'], ax=ax,
# 			#	                    title="Predictions")
# 			#log("gt_class_id", gt_class_id)
# 			#log("gt_bbox", gt_bbox)
# 			#log("gt_mask", gt_mask)

# 			#splash = sclera.color_splash(image, r['masks'])
# 			#display_images([splash], cols=1)
# 		except:
# 			print("Failed val image :: " + img)
# 			logfile.write("%s\n" % img)
# 			continue
# 	else:
# 		print("Failed val image :: " + img)
# 		logfile.write("%s\n" % img)
# 	image_id =  image_id + 1

# image_id = 0
# for img in glob.glob("/home/user1/Desktop/kiran-sandbox/Mask_RCNN/samples/Sclera/dataset/train/*.jpg"):
# 	print(img)
# 	image = skimage.io.imread(img)
# 	#source_file_name = "mask-input/source-" + str(image_id) + ".jpg"
# 	source_file_name = "resnet50-mask-output-train-set/source-" + img.split('/')[-1]
# 	skimage.io.imsave(source_file_name, image)
# 	# Run object detection
# 	results = model.detect([image], verbose=1)[0]
# 	mask_image = results['masks']
# 	if mask_image.size != 0:
# 		try:
# 			print(mask_image.shape)
# 			splash = sclera.color_splash(image, results['masks'])
# 			# Save output
# 			splash_file_name = "resnet50-mask-output-train-set/splash-" + img.split('/')[-1]# str(image_id) + ".jpg"
# 			skimage.io.imsave(splash_file_name, splash)


# 			mask_file_name = "resnet50-mask-output-train-set/mask-" + img.split('/')[-1] #str(image_id) + ".jpg"
# 			skimage.io.imsave(mask_file_name, img_as_uint(mask_image))
				
# 			# Display results
# 			#ax = get_ax(1)
# 			#r = results
# 			#visualize.display_instances(image, results['rois'], results['masks'], results['class_ids'], 
# 			#	                    dataset.class_names, results['scores'], ax=ax,
# 			#	                    title="Predictions")
# 			#log("gt_class_id", gt_class_id)
# 			#log("gt_bbox", gt_bbox)
# 			#log("gt_mask", gt_mask)

# 			#splash = sclera.color_splash(image, r['masks'])
# 			#display_images([splash], cols=1)
# 		except:
# 			print("Failed train image :: " + img)
# 			continue
# 	else:
# 		print("Failed train/image :: " + img)
# 	image_id =  image_id + 1
