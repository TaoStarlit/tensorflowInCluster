# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import Z99Model as mymodel

FLAGS = None



def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
  
  print("using VGG5GAP_REGUL");
  y_conv, regularizer = mymodel.VGG5GAP_Regul(x_image)
  #beta=0.001 # train and test 0.96    step 1100, training accuracy 0.96, total_loss 0.123597, regularizer 0.018564
  #beta=0.5 # train and test 98 0.966  step 1100, training accuracy 0.98, total_loss 2.64369, entroy_loss 0.115346 regularizer 2.52834
  beta=1 # 0.96 0.96 step 1100, training accuracy 0.96, total_loss 3.45786, entroy_loss 0.132352 regularizer 3.32551
  beta = 100 #  0.94 0.95 step 1100, training accuracy 0.94, total_loss 230.503, entroy_loss 0.183427 regularizer 230.319


  print('lambda of regularization %g'%(beta))

  with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
    total_loss = cross_entropy + beta*regularizer

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  #graph_location = tempfile.mkdtemp()#Saving graph to: C:\Users\zheng\AppData\Local\Temp\tmphap3mpy_
  graph_location = 'D:/tmp/10.1VGG11/'
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1200): #20000 1200 500
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy, loss, cross_entropy_value, regularizer_part=sess.run([accuracy,total_loss,cross_entropy,regularizer],feed_dict={x: batch[0], y_: batch[1]})
        print('step %d, training accuracy %g, total_loss %g, entroy_loss %g regularizer %g' % (i, train_accuracy, loss, cross_entropy_value, regularizer_part*beta))

      train_step.run(feed_dict={x: batch[0], y_: batch[1]})
      
      if i%400==0:
        test_accuracy, loss, regularizer_part = sess.run([accuracy,total_loss,regularizer],feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print('step %d, test accuracy %g, total_loss %g, regularizer %g' % (i, test_accuracy, loss, regularizer_part * beta))

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='D:/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)