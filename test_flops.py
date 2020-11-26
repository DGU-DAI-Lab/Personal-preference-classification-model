import tensorflow as tf
import os
import cv2
import numpy as np
from keras.utils import np_utils
import glob
from scipy.misc import imread, imresize
from keras.datasets import mnist
import keras.backend as K


#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#medel definiion: Convolutional neural network

def load_image():
	print("Loading image")
	paths = glob.glob("./a/*.jpg")
	num_img= len(paths)
	b_img = np.empty((1,224,224,3),int)
	for j in paths:
		n = cv2.imread(j)/.255
		n = imresize(n, (224, 224))
		#b_img = np.array(b_img)
		n = np.expand_dims(n, axis=0)
		print(n.shape)
		b_img=np.append(b_img,n, axis=0)


def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    #print(len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def build_CNN_classifier(x):

  x_image = tf.reshape(x, [-1, 28, 28, 1])
""""""
 # w_trans_conv1=tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 1],dtype=tf.float32, stddev=5e-2))
  #b_trans_conv1=tf.Variable(tf.constant(0.1,dtype=tf.float32, shape=[1]))
  #c_pyramid1=tf.layers.relu(tf.nn.conv2d_transpose(x_image, w_trans_conv1, output_shape=[-1,32,32,1], strides=[1, 1, 1, 1], padding='SAME'))# 2/2

  W_conv1 = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 1], dtype=tf.float32, stddev=5e-2))
  b_conv1 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[1]))
  c_pyramid0 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)#28
  c_pyramid1 = tf.nn.relu(tf.layers.conv2d_transpose(x_image, 1, [2, 2], strides=(2, 2), padding='SAME'))#56
  c_pyramid2 = tf.nn.relu(tf.layers.conv2d_transpose(x_image, 1, [3, 3], strides=(3, 3), padding='SAME'))#84
  c_pyramid3 = tf.nn.relu(tf.layers.conv2d_transpose(x_image, 1, [4, 4], strides=(4, 4), padding='SAME'))#112
  #print(c_pyramid2.numpy())
  c_pool0=tf.nn.max_pool(c_pyramid3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')#56
  c_pool1=tf.nn.max_pool(c_pyramid2, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')#28
  print(c_pool0)
  c_sub0=tf.abs((tf.subtract(c_pyramid0, c_pool1))) #|28-28|
  c_sub1 = tf.abs((tf.subtract(c_pyramid1, c_pool0)))  # |56-56|
  c_sub1=tf.nn.max_pool(c_sub1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # |28-28|
  c_0=tf.add(c_sub0,c_sub1)

 # w_trans_conv2=tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 1],dtype=tf.float32, stddev=5e-2))
 # b_trans_conv2=tf.Variable(tf.constant(0.1,dtype=tf.float32, shape=[1]))
  #c_pyramid2=  tf.nn.relu(tf.nn.conv2d_transpose(x_image, w_trans_conv2, output_shape=[-1,42,42,1], strides=[1, 1, 1, 1], padding='SAME')+b_trans_conv2)#3/2



  #w_trans_conv3 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 1], dtype=tf.float32, stddev=5e-2))
  #b_trans_conv3 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[1]))
  #c_pyramid3= tf.nn.relu(tf.nn.conv2d_transpose(x_image, w_trans_conv3, output_shape=[-1, 56, 56, 1], strides=[1, 1, 1, 1],padding='SAME') + b_trans_conv3)# 4/2
  #c_pool1 = tf.layers.max_pooling2d(c_pyramid3, [2,2], [2,2],padding='SAME')

  #w_trans_conv4 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 1],dtype=tf.float32,  stddev=5e-2))
  #b_trans_conv4 = tf.Variable(tf.constant(0.1,dtype=tf.float32,  shape=[1]))
  #c_pyramid4=tf.nn.relu(tf.nn.conv2d_transpose(x_image, w_trans_conv4, output_shape=[-1, 84, 84, 1], strides=[1, 1, 1, 1], padding='SAME') + b_trans_conv4)#6/2
  #c_pool2 =tf.layers.max_pooling2d(c_pyramid4, [2,2], [2,2],padding='SAME')


  #subtract1=tf.subtract(c_pyramid1,c_pool1)#28


 # subtract2=tf.subtract(c_pyramid2,c_pool2 )#42(3/2)
#
 # w_trans_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 1],dtype=tf.float32,  stddev=5e-2))
  #b_trans_conv5 = tf.Variable(tf.constant(0.1,dtype=tf.float32,  shape=[1]))
  #c_resize = tf.nn.relu(tf.nn.conv2d_transpose(subtract1, w_trans_conv5, output_shape=[1, 42, 42, 1], strides=[1, 1, 1, 1],padding='SAME') + b_trans_conv5)  #48( 3/2)

 # goal_1=tf.nn.relu(tf.add(subtract2,c_resize))#42+42
 # print(goal_1.shape)

  W_conv1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 32], dtype=tf.float32, stddev=5e-2))
  b_conv1 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[32]))
  h_conv1 = tf.nn.relu(tf.nn.conv2d(c_0, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

  W_conv2 = tf.Variable(tf.truncated_normal(shape=[1, 1, 32, 64],dtype=tf.float32,  stddev=5e-2))
  b_conv2 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[64]))
  h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2,strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
  #h_drop1 = tf.nn.dropout(h_conv3, rate=0.2)
  h_pool1=tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  W_conv3 = tf.Variable(tf.truncated_normal(shape=[1, 1, 64, 128],dtype=tf.float32,  stddev=5e-2))
  b_conv3 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[128]))
  h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)


  h_pool2 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  h_drop1 = tf.nn.dropout(h_pool2, rate=0.2)

  W_fc1 = tf.Variable(tf.truncated_normal(shape=[7 * 7 * 128, 1024],dtype=tf.float32,  stddev=5e-2))
  b_fc1 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[1024]))
  h_pool2_flat = tf.reshape(h_drop1, [-1, 7 * 7 *128])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  h_drop2 = tf.nn.dropout(h_fc1 , rate=0.2)
  # Output Layer

  W_output = tf.Variable(tf.truncated_normal(shape=[1024, 10],dtype=tf.float32,  stddev=5e-2))
  b_output = tf.Variable(tf.constant(0.1,dtype=tf.float32,  shape=[10]))
  logits = tf.matmul(h_drop2, W_output) + b_output
  y_pred = tf.nn.softmax(logits)

  return y_pred, logits, c_pyramid2

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
#y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)
#y_test_one_hot =y_test
#y_train_one_hot=y_train
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
y_train_one_hot = np_utils.to_categorical(y_train)
y_test_one_hot = np_utils.to_categorical(y_test)
x = tf.placeholder(tf.float32, shape=[1, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

# Convolutional Neural Networks(CNN)을 선언합니다.
y_pred, logits,c_pyramid2 = build_CNN_classifier(x)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)


correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# tf.train.Saver
SAVER_DIR = "model"
saver = tf.train.Saver()
checkpoint_path = os.path.join(SAVER_DIR, "model")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

run_meta = tf.RunMetadata()
with tf.Session(graph=tf.Graph()) as sess:
    K.set_session(sess)
    net = build_CNN_classifier(x)

    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))


