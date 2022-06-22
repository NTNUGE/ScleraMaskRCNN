from PIL import Image
import glob
read_path = glob.glob("/home/user1/Desktop/GOURAV/MagFace/MagFace-main/inference/dataset/class2/*")
newsize = (112, 112) 
for get_image in read_path:
   im = Image.open(get_image)
   im1 = im.resize(newsize) 
   get_data = get_image.split("/")[-1]
   print(get_data)
   get_data = get_data.replace(".png",".jpg")
   print(get_data)
   #rgb_im = im.convert('RGB')
   im1.save(r"/home/user1/Desktop/GOURAV/MagFace/MagFace-main/inference/dataset/class2_v1/" + get_data)