from skimage import io, color
#from natsort import natsorted, ns
from skimage import data
from skimage.color import rgb2hsv
from skimage.color import rgb2ycbcr
import numpy as np 
import glob
from PIL import Image  
import PIL 
import cv2
import os
read_path = glob.glob(r"C:\Users\goura\Downloads\MSD\\" + '\\**\\*.jpg',recursive=True)
print(read_path)
for get_image in read_path: 
 print(get_image)
 get_image1 = get_image.split("\\")[-1]
 print(get_image1)
 rgb = io.imread(get_image)/255
 lab = color.rgb2lab(rgb)


 #for RGB-HSV
 hsv_img = rgb2hsv(rgb)

 #for RGB-Ycbcr
 get_ycbcr = rgb2ycbcr(rgb)

 concat = hsv_img/255+get_ycbcr/255+lab/255

 print(concat)
 #io.imshow(concat)
 #io.show()

 print(concat)
 io.imsave(r"C:\Users\goura\OneDrive\Desktop\SSBCDATASET\output\\" + get_image1, concat)
 #concat.save(result_image_name, "PNG", optimize=True)
 #cv2.imwrite(r"C:\Users\goura\OneDrive\Desktop\SSBCDATASET\output\\" + get_image1 + ".jpg", concat/255) 



# im1 = concat.save(r"C:\Users\goura\OneDrive\Desktop\SSBCDATASET\output\\" + get_image1 + ".jpg") 

 #io.imshow(concat) 
 #io.show()
