import cv2
import glob
import numpy as np 
import csv


read_path = glob.glob(r"/home/user1/Desktop/kiran-sandbox/Mask_RCNN/samples/Sclera/dataset/predict_test/segmentator/ALL/MOBIUS/Binarised/*.png") #+ '//**//*.jpg',recursive=True)

bp = 0
wp = 0
all_count = 0

for get_data in read_path:
	image = cv2.imread(get_data)
	gray_version = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	Total_pixel = 480*320
	print('TP:', Total_pixel)
	n_white_pix = np.sum(gray_version == 255)
	n_black_pix = np.sum(gray_version == 0)
	m_wb = np.sum(gray_version > 0) 
	print('Number of white pixels:', n_white_pix)
	print('Number of black pixels:', n_black_pix)
	pix_sum = n_white_pix + n_black_pix
	if Total_pixel == pix_sum:
		print("Binarised")
		bp = bp + 1
	elif m_wb == True:
		all_count = all_count + 1

	else:
		print("Image is not binarised")
		wp = wp + 1
		myFile = open('./Non-binarised.txt', 'a')
		myFile = myFile.write(get_data)
print(bp)
print(wp)
print(all_count)
#myFile.close()
