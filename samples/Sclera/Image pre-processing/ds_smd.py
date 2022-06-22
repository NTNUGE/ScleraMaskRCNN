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
from skimage import io, color
#from natsort import natsorted, ns
from skimage import data
from skimage.color import rgb2hsv
from skimage.color import rgb2ycbcr
newsize = (480, 320) 

result_dir = r'/home/user1/Desktop/kiran-sandbox/Mask_RCNN/samples/Sclera/dataset/predict_test/ALL/SMD/new_pred//'


for folderName, subfolders, filenames in os.walk('/home/user1/Desktop/kiran-sandbox/Mask_RCNN/samples/Sclera/dataset/predict_test/ALL/SMD/Predictions/'):
    print('The current folder is ' + folderName + str(filenames))
    folderName1 = folderName.split('/')[-1]
    folderName1 = str(folderName1)
    #print(folderName1)
    mask_result_dir = result_dir + str(folderName1)
    #mask_result_dir = result_dir + str(folderName1
    if not os.path.exists(mask_result_dir):
    	os.mkdir(mask_result_dir)
    for get_image in filenames:
        print(get_image)
        get_image1 = get_image.split("/")[-1]
        print(get_image1)
        img = cv2.imread(folderName + '/' + get_image,0)
        #img = skimage.color.gray2rgb(skimage.color.rgb2gray(img))
        #img = cv2.imread(folderName + '/' + get_image)
        resized_image = cv2.resize(img, (480, 320))
        result = cv2.imread("/home/user1/Desktop/kiran-sandbox/Mask_RCNN/samples/Sclera/dataset/predict_test/5.png",0)
        x, y, w, h = 300, 300, 300,300
        ROI = result[y:y+h, x:x+w]
        print(ROI)
        mean, STD  = cv2.meanStdDev(ROI*3)
        #image1 = cv2.imread(splash_file_name,0)
        offset = 0.7
        clipped = np.clip(img, mean - offset*STD, mean + offset*STD).astype(np.uint8)
        get_normalize = cv2.normalize(clipped, clipped, 0, 255, norm_type=cv2.NORM_MINMAX)
        print("get_image:",get_image1)
        #get_image1 = get_image1.replace(".jpg", ".png")
        #get_image1 = get_image.split(".")[-2]
        io.imsave(mask_result_dir + '/' + get_image1, get_normalize)
# Plotting of source and destination image 
#plt.subplot(121), plt.imshow(img) 
#plt.subplot(122), plt.imshow(concat/255) 
  
#plt.show() 