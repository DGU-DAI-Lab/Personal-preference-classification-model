"""
Each variable stands for
    w: weights
    b: biases
    h: hidden layer
    l: logits
        of the layer which they belong to.
"""

import tensorflow as tf

def CNN(x, batch_prob=tf.placeholder(tf.bool)):
    x = tf.reshape(x, [-1, 32, 32, 1])

    # Block 1

    with tf.name_scope("Conv. Layer 1"):
        w = tf.Variable(tf.truncated_normal(shape=[7, 7, 1, 128], stddev=5e-2), name='w1')
        b = tf.Variable(tf.constant(0.1, shape=[128]), name='b1')
        h = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
        h = tf.layers.batch_normalization(h, center=True, scale=True, training=batch_prob, name='bn1')
        x = tf.nn.relu(h)

    tf.add_to_collection('w1', w)
    tf.add_to_collection('b1', b)
    tf.add_to_collection('bn1', h)

    with tf.name_scope("Conv. Layer 2"):
        w = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 128], stddev=5e-2), name='w2')
        b = tf.Variable(tf.constant(0.1, shape=[128]), name='b2')
        h = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
        h = tf.layers.batch_normalization(h, center=True, scale=True, training=batch_prob, name='bn2')
        x = tf.nn.relu(h)

    tf.add_to_collection('w2', w)
    tf.add_to_collection('b2', b)
    tf.add_to_collection('bn2', h)

    with tf.name_scope("Conv. Layer 3 (MaxPool, Dropout)"):
        w = tf.Variable(tf.truncated_normal(shape=[1, 1, 128, 64], stddev=5e-2), name='w3')
        b = tf.Variable(tf.constant(0.1, shape=[64]), name='b3')
        h = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
        h = tf.layers.batch_normalization(h, center=True, scale=True, training=batch_prob, name='bn3')
        x = tf.nn.relu(h)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        x = tf.nn.dropout(x, rate=0.4)
    
    # Block 2

    with tf.name_scope("Conv. Layer 4"):
        w = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=5e-2), name='w4')
        b = tf.Variable(tf.constant(0.1, shape=[64]), name='b4')
        h = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
        h = tf.layers.batch_normalization(h, center=True, scale=True, training=batch_prob, name='bn4')
        x = tf.nn.relu(x)

    with tf.name_scope("Conv. Layer 5"):
        w = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=5e-2), name='w5')
        b = tf.Variable(tf.constant(0.1, shape=[64]), name='b5')
        h = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
        h = tf.layers.batch_normalization(h, center=True, scale=True, training=batch_prob, name='bn5')
        x = tf.nn.relu(x)

    with tf.name_scope("Conv. Layer 6 (MaxPool, Dropout)"):
        w = tf.Variable(tf.truncated_normal(shape=[1, 1, 64, 32], stddev=5e-2), name='w6')
        b = tf.Variable(tf.constant(0.1, shape=[32]), name='b6')
        h = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
        h = tf.layers.batch_normalization(h, center=True, scale=True, training=batch_prob, name='bn6')
        x = tf.nn.relu(h)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        x = tf.nn.dropout(x, rate=0.4)

    # Block 3

    with tf.name_scope("Conv. Layer 7"):
        w = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], stddev=5e-2), name='w7')
        b = tf.Variable(tf.constant(0.1, shape=[32]), name='b7')
        h = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
        h = tf.layers.batch_normalization(h, center=True, scale=True, training=batch_prob, name='bn7')
        x = tf.nn.relu(h)

    with tf.name_scope("Conv. Layer 8"):
        w = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], stddev=5e-2), name='w8')
        b = tf.Variable(tf.constant(0.1, shape=[32]), name='b8')
        h = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
        h = tf.layers.batch_normalization(h, center=True, scale=True, training=batch_prob, name='bn8')
        x = tf.nn.relu(h)

    with tf.name_scope("Conv. Layer 9 (MaxPool, Dropout)"):
        w = tf.Variable(tf.truncated_normal(shape=[1, 1, 32, 32], stddev=5e-2), name='w9')
        b = tf.Variable(tf.constant(0.1, shape=[32]), name='b9')
        h = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
        h = tf.layers.batch_normalization(h, center=True, scale=True, training=batch_prob, name='bn9')
        x = tf.nn.relu(h)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        x = tf.nn.dropout(x, rate=0.4)
    
    x = tf.reshape(x, [-1, 28 * 28 * 128])

    with tf.name_scope("Dense 1"):
        w = tf.Variable(tf.truncated_normal(shape=[4 * 4 * 32, 256], stddev=5e-2))
        b = tf.Variable(tf.constant(0.1, shape=[6096]))
        h = tf.matmul(x, w) + b
        x = tf.nn.relu(h)

    # Output

    with tf.name_scope("Dense 2"):
        w = tf.Variable(tf.truncated_normal(shape=[256, 2], stddev=5e-2))
        b = tf.Variable(tf.constant(0.1, shape=[2]))
        h = tf.matmul(x, w) + b
        l = tf.nn.softmax(h)

    return l
