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
#newsize = (480, 320) 

read_path = glob.glob(r"/home/user1/Desktop/kiran-sandbox/Mask_RCNN/samples/Sclera/dataset/NIR/nir_submission/testing_nir/CASIA-Iris-Mobile-V1.0/test/image/*")
#print(read_path)
for get_image in read_path: 
	print(get_image)
	get_image1 = get_image.split("/")[-1]
	get_image1 = get_image1.replace(".jpg",".png")
	print(get_image1)
  
# Reading image from folder where it is stored 
	img = cv2.imread(get_image) 
	alpha = 1.5 # Contrast control (1.0-3.0)
	beta = 0 # Brightness control (0-100)

#adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

#dst = cv2.fastNlMeansDenoisingColored(adjusted,None,10,10,7,21) 

#img1 = cv2.imread('Cybertruck.jpg',1)
# Creating our sharpening filter
#blur = cv2.bilateralFilter(image,9,75,75)
	filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# Applying cv2.filter2D function on our Cybertruck image
	dst = cv2.filter2D(img,-1,filter)
#cv2.imshow('blur',sharpen_img_1)


	#dst = cv2.fastNlMeansDenoisingColored(dst,None,10,10,7,21) 
	adjusted = cv2.convertScaleAbs(dst, alpha=alpha, beta=beta)
	#resized_image = cv2.resize(adjusted, (480, 320)) 

#cv2.imshow('detected circles',resized_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

	io.imsave(r"/home/user1/Desktop/kiran-sandbox/Mask_RCNN/samples/Sclera/dataset/NIR/nir_submission/testing_nir/CASIA-Iris-Mobile-V1.0/test/image_v1/" + get_image1, adjusted)
# Plotting of source and destination image 
#plt.subplot(121), plt.imshow(img) 
#plt.subplot(122), plt.imshow(concat/255) 
  
#plt.show() 