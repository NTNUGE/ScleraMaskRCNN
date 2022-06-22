from skimage import img_as_uint, io
import os
import cv2
import sys 
import glob
from PIL import Image
import numpy as np

result_dir = sys.argv[1]
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

boundary_directory = result_dir + "\\Inner_Boundary\\"
if not os.path.exists(boundary_directory):
    os.mkdir(boundary_directory)



read_path = glob.glob(r"C:\Users\goura\OneDrive\Desktop\SSBCDATASET\NIR\v1\CASIA-Iris-Africa\Iris_outer\SegmentationClass\*")

for get_image in read_path: 
	get_image2 = cv2.imread(get_image, cv2.IMREAD_COLOR)
	print(get_image)
	get_image1 = get_image.split("\\")[-1]
	print(get_image1)

	get_image2 = cv2.medianBlur(get_image2,5)
	# Convert to grayscale. 
	gray = cv2.cvtColor(get_image2, cv2.COLOR_BGR2GRAY)
	detected_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 80, param1 = 300, param2 = 20, minRadius = 10, maxRadius = 300) 
	print("Get the array size of detected circles", detected_circles)

	# Draw circles that are detected. 
	if detected_circles is not None:
		detected_circles = np.uint16(np.around(detected_circles)) 

		for pt in detected_circles[0, :]: 
			a, b, r = pt[0], pt[1], pt[2] 

			# Draw the circumference of the circle. 
			light_orange = (100, 19, 214)
			dark = (100, 20, 215)

			get_circle = cv2.circle(get_image2, (a, b), r, (100, 20, 215), 1) 
			get_circle = cv2.inRange(get_circle, light_orange, dark)
			print(get_circle)
			cv2.imwrite(boundary_directory+ get_image1 + '.png', get_circle)


