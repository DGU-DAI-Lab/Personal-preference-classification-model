import tensorflow as tf
import numpy as np
from skimage.transform import resize

def grad_cam(x, y_pred, sess, predicted_class, h_pool3, nb_classes,x_placeholder):
	x = np.expand_dims(x, axis=0)
	print("Setting gradients to 1 for target class and rest to 0")
	# Conv layer tensor [?,7,7,512]
	conv_layer = h_pool3
	# [1000]-D tensor with target class index set to 1 and rest as 0
	one_hot = tf.sparse_to_dense(predicted_class, [nb_classes], 1.0)
	signal = tf.multiply(y_pred, one_hot)
	loss = tf.reduce_mean(signal)

	grads = tf.gradients(loss, conv_layer)[0]
	# Normalizing the gradients
	norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

	output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={x_placeholder: x})
	output = output[0]           # [7,7,512]
	grads_val = grads_val[0]	 # [7,7,512]

	weights = np.mean(grads_val, axis = (0, 1)) 			# [512]
	cam = np.ones(output.shape[0 : 2], dtype = np.float32)	# [7,7]

	# Taking a weighted average
	for i, w in enumerate(weights):
	    cam += w * output[:, :, i]

	# Passing through ReLU
	cam = np.maximum(cam, 0)
	cam = cam / np.max(cam)
	cam = resize(cam, (224,224))

	# Converting grayscale to 3-D
	cam3 = np.expand_dims(cam, axis=2)
	cam3 = np.tile(cam3,[1,1,3])

	return cam3