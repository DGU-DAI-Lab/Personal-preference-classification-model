import tensorflow as tf

def CNN(x,batch_prob):
    gamma = 0.01
   # batch_prob = tf.placeholder(tf.bool)

    x_image = tf.reshape(x, [-1, 32, 32, 1])
    with tf.name_scope("convolutional_base"):

      #block1
        W_conv1 = tf.Variable(tf.truncated_normal(shape=[7, 7, 1, 128], stddev=5e-2),name='w1')
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[128]),name='b1')
        h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
        h_conv1 = tf.layers.batch_normalization(h_conv1, center=True, scale=True, training=batch_prob, name='bn1')
        h_relu1=tf.nn.relu(h_conv1)

        l2_loss_W = gamma * tf.nn.l2_loss(W_conv1)

        W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 128], stddev=5e-2),name='w2')
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[128]),name='b2')
        h_conv2 = tf.nn.conv2d(h_relu1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
        h_conv2 = tf.layers.batch_normalization(h_conv2, center=True, scale=True, training=batch_prob,name='bn2')
        h_relu2 = tf.nn.relu(h_conv2)

        l2_loss_W = + gamma * tf.nn.l2_loss(W_conv2)

        W_conv3 = tf.Variable(tf.truncated_normal(shape=[1, 1, 128, 64], stddev=5e-2),name='w3')
        b_conv3 = tf.Variable(tf.constant(0.1, shape=[64]),name='b2')
        h_conv3 = tf.nn.conv2d(h_relu2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
        h_conv3 = tf.layers.batch_normalization(h_conv3, center=True, scale=True, training=batch_prob,name='bn2')
        h_relu3 = tf.nn.relu(h_conv3)

        l2_loss_W = + gamma * tf.nn.l2_loss(W_conv3)

        h_pool1 = tf.nn.max_pool(h_relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h_drop1=tf.nn.dropout(h_pool1, rate=0.4)

        #block 2
        W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=5e-2),name='w4')
        b_conv4 = tf.Variable(tf.constant(0.1, shape=[64]),name='b4')
        h_conv4 = tf.nn.conv2d(h_drop1, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
        h_conv4 = tf.layers.batch_normalization(h_conv4, center=True, scale=True, training=batch_prob,name='bn4')
        h_relu4 = tf.nn.relu(h_conv4)

        l2_loss_W = + gamma * tf.nn.l2_loss(W_conv4)

        W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=5e-2),name='w5')
        b_conv5 = tf.Variable(tf.constant(0.1, shape=[64]),name='b5')
        h_conv5 = tf.nn.conv2d(h_relu4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
        h_conv5 = tf.layers.batch_normalization(h_conv5, center=True, scale=True, training=batch_prob,name='bn5')
        h_relu5 = tf.nn.relu(h_conv5)
        l2_loss_W = + gamma * tf.nn.l2_loss(W_conv5)

        W_conv6 = tf.Variable(tf.truncated_normal(shape=[1, 1, 64, 32], stddev=5e-2),name='w6')
        b_conv6 = tf.Variable(tf.constant(0.1, shape=[32]),name='b6')
        h_conv6 = tf.nn.conv2d(h_relu5, W_conv6, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
        h_conv6 = tf.layers.batch_normalization(h_conv6, center=True, scale=True, training=batch_prob,name='bn6')
        h_relu6 = tf.nn.relu(h_conv6)
        l2_loss_W = + gamma * tf.nn.l2_loss(W_conv6)

        h_pool2 = tf.nn.max_pool(h_relu6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h_drop2=tf.nn.dropout(h_pool2, rate=0.4)

        #block3

        W_conv7 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], stddev=5e-2),name='w7')
        b_conv7 = tf.Variable(tf.constant(0.1, shape=[32]),name='b7')
        h_conv7 = tf.nn.conv2d(h_drop2, W_conv7, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
        h_conv7 = tf.layers.batch_normalization(h_conv7, center=True, scale=True, training=batch_prob,name='bn7')
        h_relu7 = tf.nn.relu(h_conv7)

        l2_loss_W = + gamma * tf.nn.l2_loss(W_conv7)

        W_conv8 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], stddev=5e-2),name='w8')
        b_conv8 = tf.Variable(tf.constant(0.1, shape=[32]),name='b8')
        h_conv8 = tf.nn.conv2d(h_relu7, W_conv8, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
        h_conv8 = tf.layers.batch_normalization(h_conv8, center=True, scale=True, training=batch_prob,name='bn8')
        h_relu8 = tf.nn.relu(h_conv8)

        l2_loss_W = + gamma * tf.nn.l2_loss(W_conv8)

        W_conv9 = tf.Variable(tf.truncated_normal(shape=[1, 1, 32, 32], stddev=5e-2),name='w9')
        b_conv9 = tf.Variable(tf.constant(0.1, shape=[32]),name='b9')
        h_conv9 = tf.nn.conv2d(h_relu8, W_conv9, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
        h_conv9 = tf.layers.batch_normalization(h_conv9, center=True, scale=True, training=batch_prob,name='bn9')
        h_relu9 = tf.nn.relu(h_conv8)

        l2_loss_W = + gamma * tf.nn.l2_loss(W_conv9)

        h_pool3 = tf.nn.max_pool(h_relu9, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h_drop3= tf.nn.dropout(h_pool3, rate=0.4)
    train_vars = {'w1': W_conv1, 'b1': b_conv1, 'bn1':h_conv1, 'w2': W_conv2, 'b2': b_conv2, 'bn1': h_conv2 }
    for key, var in train_vars.items():
        tf.add_to_collection(key, var)

    with tf.name_scope("classifier"):

        W_fc1 = tf.Variable(tf.truncated_normal(shape=[4 * 4 * 32, 256], stddev=5e-2))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[6096]))
        h_pool2_flat = tf.reshape(h_drop3, [-1, 28 * 28 * 128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        l2_loss_W = + gamma * tf.nn.l2_loss(W_fc1)

        # Output Layer

        W_output = tf.Variable(tf.truncated_normal(shape=[256, 2], stddev=5e-2))
        b_output = tf.Variable(tf.constant(0.1, shape=[2]))
        logits = tf.matmul(h_fc1, W_output) + b_output
        y_pred = tf.nn.softmax(logits)

        l2_loss_W = + gamma * tf.nn.l2_loss(W_output)



    return y_pred, logits, l2_loss_W,h_pool3,train_vars