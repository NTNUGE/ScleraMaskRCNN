import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
from skimage import io, color
#from natsort import natsorted, ns
from skimage import data
from skimage.color import rgb2hsv
from skimage.color import rgb2ycbcr
import numpy as np 
import glob
import os 
from PIL import Image

import cv2
newsize = (480, 320) 

read_path = glob.glob(r"/home/user1/Desktop/kiran-sandbox/Mask_RCNN/samples/Sclera/dataset/test/ssbc-2020-challenge-dataset-test/SSBC-2020-MOBIUS-resized-contrast-adjusted/*.jpg") #+ '//**//*.jpg',recursive=True)
###

print(read_path)
for get_image in read_path: 
		print(get_image)
		get_image1 = get_image.split("/")[-1]
		print(get_image1)
	  
		img = cv2.imread(get_image)#(r"C:\Windows\System32\Mask_RCNN\dataset\16.jpg") 
        
		RGB = color.lab2rgb(img)

		alpha = 1.5
		beta = 0

		#dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21) 
		resized_image = cv2.resize(img, (480, 320)) 
		adjusted = cv2.convertScaleAbs(resized_image, alpha=alpha, beta=beta)
		#filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
		#sharpen_img_1=cv2.filter2D(resized_image,-1,filter)
		io.imsave("/home/user1/Desktop/kiran-sandbox/Mask_RCNN/samples/Sclera/dataset/SSBC_FINAL/test-dataset/MOBIUS/orginal/" + get_image1,adjusted)#sharpen_img_1)