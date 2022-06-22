import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, color
#from natsort import natsorted, ns
from skimage import data
from skimage.color import rgb2hsv
from skimage.color import rgb2ycbcr
import numpy as np 
import glob
from PIL import Image
import glob
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


#Global Variables
kernel = np.array([[100, -1, 0], 
                   [-1, 100,-1], 
                   [0, -1, 100]])

read_path = glob.glob(r"/home/user1/Desktop/kiran-sandbox/Mask_RCNN/samples/Sclera/dataset/NIR/nir_submission/testing_nir/CASIA-Iris-Asia/CASIA-distance/test/image/*")

for get_image in read_path:
   im = cv2.imread(get_image,cv2.IMREAD_GRAYSCALE)
   #im = cv2.fastNlMeansDenoisingColored(im,None,10,10,7,21)
   #im1 = im.resize(newsize) 
   # keeping the image name and extension Name
   get_data = get_image.split("/")[-1]
   get_data = get_data.split(".")[-2]
   ret, thresh3 = cv2.threshold(im, 90, 250, cv2.THRESH_TRUNC)
   image_sharp = cv2.filter2D(thresh3, -4, kernel)

   #ret, thresh3 = cv2.threshold(image_sharp, 120, 255, cv2.THRESH_TRUNC)
   print(get_data)
   #rgb_im = im1.convert('RGB')
   cv2.imwrite(r"/home/user1/Desktop/kiran-sandbox/Mask_RCNN/samples/Sclera/dataset/NIR/nir_submission/testing_nir/CASIA-Iris-Asia/CASIA-distance/test/image_v2/" + get_data + '.png', thresh3)
   #thresh3.save(r"C:\Users\goura\OneDrive\Desktop\SSBCDATASET\NIR\NIR_NEWDATASET\\" + get_data + '.png')



#image = cv2.imread(r'C:\Users\goura\OneDrive\Desktop\SSBCDATASET\NIR\train\CASIA_benchmark_African_type2_52.jpg', cv2.IMREAD_GRAYSCALE)
# Create kernel


# Sharpen image

#thresh3= cv2.medianBlur(image_sharp,1)
#ret, thresh3 = cv2.threshold(thresh3, 150, 255, cv2.THRESH_TRUNC)


#thresh3 = cv2.Canny(thresh3,150,200)

#thresh3 = thresh3[thresh3 == 255] = [0,0,255]
#thresh3 = cv2.filter2D(thresh3, -1, kernel)
'''
alpha = 3.0 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)
filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# Applying cv2.filter2D function on our Cybertruck image
dst = cv2.filter2D(thresh3,-1,filter)
thresh3 = cv2.convertScaleAbs(dst, alpha=alpha, beta=beta)
#thresh3 = cv2.filter2D(thresh3, -1, kernel)

#thresh3 = cv2.cvtColor(thresh3,cv2.COLOR_GRAY2RGB)
#thresh3 = cv2.filter2D(thresh3, -1, kernel)

#thresh3 = thresh3[thresh3 == 255] = [0, 0, 255]
#thresh3 = cv2.cvtColor(thresh3, cv2.COLOR_BGR2GRAY) 
'''
'''
circles = cv2.HoughCircles(thresh3,cv2.HOUGH_GRADIENT,1,10,
                            param1=10,param2=40,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(thresh3,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(thresh3,(i[0],i[1]),2,(0,0,255),3)

cv2.imwrite('new_image.png', thresh3)
'''

#cv2.imshow('detected circles',image_sharp)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
