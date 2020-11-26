import cv2
import glob
from scipy.misc import imread, imresize
import numpy as np

def load_image(a):
	print("Loading train image")
	paths = glob.glob(a)
	num_img= len(paths)
	b_img = np.empty((1,224,224,3),int)
	for j in paths:
		n = cv2.imread(j)/.255
		n = imresize(n, (224, 224))
		#b_img = np.array(b_img)
		n = np.expand_dims(n, axis=0)
		print(n.shape)
		b_img=np.append(b_img,n, axis=0)
		print(b_img.shape)
		return b_img, num_img