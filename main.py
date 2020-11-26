from typing import List, Any

from grad_cam import grad_cam
from CNN import CNN
from load_image import load_image
import tensorflow as tf
import numpy as np
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
from imagenet_classes import class_names
from scipy.misc import imread, imresize
import cv2
import glob
import tensorflow as tf





def main(_):
	x ,num_img= load_image()
	sess=tf.Session()
	print("\nLoading CNN")
	imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
	y_pred, logits, l2_loss_W,h_pool3=CNN(imgs, False)

	print("\nFeedforwarding")
	prob = sess.run(y_pred, feed_dict={imgs: x})
	preds = (np.argsort(prob)[::-1])[0:5]
	#Target class
	predicted_class = preds[0]
	layer_name = "FC2"
	nb_classes = 2
	for i in range(num_img):
		#out1 = []
		cam3 = grad_cam(x[i], y_pred, sess, predicted_class[i], layer_name, nb_classes,imgs)
		heatmap = cv2.applyColorMap(np.uint8(255*cam3), cv2.COLORMAP_JET)
		out1=0.5* heatmap + 0.5* x[i]
		out1 /= out1.max()
		out1=255*out1
		# cv2.imshow('grad_cam',out1)
		k=str(i)
		cv2.imwrite("./e/" + k + "_heat.jpg",heatmap)
		cv2.imwrite("./e/" + k + "_out.jpg",out1)

if __name__ == '__main__':
	tf.app.run()

