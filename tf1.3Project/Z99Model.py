import tensorflow as tf

"""conv2d returns a 2d convolution layer with full stride."""
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

"""max_pool_2x2 downsamples a feature map by 2X."""    
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1], padding='SAME')
"""weight_variable generates a weight variable of a given shape."""
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
  """bias_variable generates a bias variable of a given shape."""
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


import math
def half_ceil(a):
  b=a/2.0
  return int(math.ceil(b))



def VGG11(x,h=28,w=28,para={1:8, 2:16, 3:32,4:32, 5:64,6:64, 7:128,8:128, 9:512,10:512,11:10}):
  """VGG11 builds the graph for a VGG11 net for classifying 2D fingerprint.
  Args:
    para: an input parameter for the number of weights in each layers
    para={1:8, 2:16, 3:32,4:32, 5:64,6:64, 7:128,8:128, 9:512,10:512:11:10}
  Returns:
    placeholder name, and output logits
  """
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([3, 3, 1, para[1]]) # the defintion can be in the latter
    b_conv1 = bias_variable([para[1]])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool_a1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps para[1] feature maps to para[21].
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3, 3, para[1], para[2]])
    b_conv2 = bias_variable([para[2]])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # pooling layer after conv2.
  with tf.name_scope('pool_a2'):
    h_pool2 = max_pool_2x2(h_conv2)


  # 3rd pooling layer.
  with tf.name_scope('conv3'):
    W_conv3 = weight_variable([3, 3, para[2], para[3]]) # the defintion can be in the latter
    b_conv3 = bias_variable([para[3]])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
  # 4rd pooling layer.
  with tf.name_scope('conv4'):
    W_conv4 = weight_variable([3, 3, para[3], para[4]]) # the defintion can be in the latter
    b_conv4 = bias_variable([para[4]])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

  # pooling layer after conv4.
  with tf.name_scope('pool_a4'):
    h_pool4 = max_pool_2x2(h_conv4)

  # 5th layer -- maps para[4] feature maps to para[5].
  with tf.name_scope('conv5'):
    W_conv5 = weight_variable([3, 3, para[4], para[5]])
    b_conv5 = bias_variable([para[5]])
    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
  # 6th layer -- maps para[5] feature maps to para[6].
  with tf.name_scope('conv6'):
    W_conv6 = weight_variable([3, 3, para[5], para[6]])
    b_conv6 = bias_variable([para[6]])
    h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

  # pooling layer after conv6.
  with tf.name_scope('pool_a6'):
    h_pool6 = max_pool_2x2(h_conv6)

  # 7th layer -- maps para[6] feature maps to para[7].
  with tf.name_scope('conv7'):
    W_conv7 = weight_variable([3, 3, para[6], para[7]])
    b_conv7 = bias_variable([para[7]])
    h_conv7 = tf.nn.relu(conv2d(h_pool6, W_conv7) + b_conv7)
  # 5th layer -- maps para[5] feature maps to para[6].
  with tf.name_scope('conv8'):
    W_conv8 = weight_variable([3, 3, para[7], para[8]])
    b_conv8 = bias_variable([para[8]])
    h_conv8 = tf.nn.relu(conv2d(h_conv7, W_conv8) + b_conv8)

  # pooling layer after conv8.
  with tf.name_scope('pool_a8'):
    h_pool8 = max_pool_2x2(h_conv8)



  # Fully connected layer No.9 -- 
  # for Mnist Expert after 2 round of downsampling, our 28x28 image is down to 7x7x64 feature maps(3136) -- maps this to 1024 features.
  # for here 28 --> 5 time pool max -- 14 --7 --4 --2 --1  1*1*128 -- 128  4time   --para9 256
  with tf.name_scope('fc9'):
    
    for i in range(5): # equlent to range(0,5,1)
       h=half_ceil(h)
       w=half_ceil(w)

    
    W_fc9 = weight_variable([h * w * para[8], para[9]]) 
    b_fc9 = bias_variable([para[9]])

    h_pool9_flat = tf.reshape(h_pool8, [-1, h * w * para[8]])
    # Input to reshape is a tensor with 6400 values, but the requested shape requires a multiple of 512
    # 6400 = 128 5 * 10
    h_fc9 = tf.nn.relu(tf.matmul(h_pool9_flat, W_fc9) + b_fc9)

  with tf.name_scope('fc10'):
    W_fc10 = weight_variable([para[9], para[10]])
    b_fc10 = bias_variable([para[10]])

    h_fc10 = tf.nn.relu(tf.matmul(h_fc9, W_fc10) + b_fc10)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc10_drop = tf.nn.dropout(h_fc10, keep_prob) ## what is co-adaptation 相互适应？？  dropout placeholder for what?


  with tf.name_scope('fc11'):
    W_fc11 = weight_variable([para[10], para[11]])
    b_fc11 = bias_variable([para[11]])

    h_fc11 = tf.matmul(h_fc10_drop, W_fc11) + b_fc11

  return h_fc11, keep_prob


# origin VGG11 image is 224*244 5 max_pool  228 -> 112 -> 56 -> 28 ->14 ->7
# here is 28, so I can only max_pool 2 times
def VGG7(x,h=28,w=28,para={1:32,2:32, 3:64,4:64, 5:512,6:512,7:10}):
  """VGG11 builds the graph for a VGG11 net for classifying 2D fingerprint.
  Args:
    para: an input parameter for the number of weights in each layers
    para={1:8, 2:16, 3:32,4:32, 5:64,6:64, 7:128,8:128, 9:512,10:512:11:10}
  Returns:
    placeholder name, and output logits
  """
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([3, 3, 1, para[1]]) # the defintion can be in the latter
    b_conv1 = bias_variable([para[1]])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

  # Second convolutional layer -- maps para[1] feature maps to para[21].
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3, 3, para[1], para[2]])
    b_conv2 = bias_variable([para[2]])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool_a2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # 3rd pooling layer.
  with tf.name_scope('conv3'):
    W_conv3 = weight_variable([3, 3, para[2], para[3]]) # the defintion can be in the latter
    b_conv3 = bias_variable([para[3]])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
  # 4rd pooling layer.
  with tf.name_scope('conv4'):
    W_conv4 = weight_variable([3, 3, para[3], para[4]]) # the defintion can be in the latter
    b_conv4 = bias_variable([para[4]])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

  # pooling layer after conv2.
  with tf.name_scope('pool_a4'):
    h_pool4 = max_pool_2x2(h_conv4)




  with tf.name_scope('fc5'):

    for i in range(2): # equlent to range(0,5,1)
       h=half_ceil(h)
       w=half_ceil(w)


    W_fc5 = weight_variable([h * w * para[4], para[5]]) 
    b_fc5 = bias_variable([para[5]])

    h_pool5_flat = tf.reshape(h_pool4, [-1, h * w * para[4]])
    # Input to reshape is a tensor with 6400 values, but the requested shape requires a multiple of 512
    # 6400 = 128 5 * 10
    h_fc5 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc5) + b_fc5)

  with tf.name_scope('fc6'):
    W_fc6 = weight_variable([para[5], para[6]])
    b_fc6 = bias_variable([para[6]])

    h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob) ## what is co-adaptation 相互适应？？  dropout placeholder for what?


  with tf.name_scope('fc7'):
    W_fc7 = weight_variable([para[6], para[7]])
    b_fc7 = bias_variable([para[7]])

    h_fc7 = tf.matmul(h_fc6_drop, W_fc7) + b_fc7 # no relu in the last layer!

  return h_fc7, keep_prob
