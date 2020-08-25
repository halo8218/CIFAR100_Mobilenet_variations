# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from input_pipeline import _get_data

class Model(object):
  """MobileNet model."""

  def __init__(self, num_classes, se1=False, se2=False):
    """MobileNet constructor.
    """
    self.num_classes = num_classes
    self.se1 = se1
    self.se2 = se2
    self._build_model()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.layers.Conv2d."""
    return (stride, stride)

  def _build_model(self):
    """Build the core model within the graph."""
    with tf.device('/cpu:0'), tf.variable_scope('input_pipeline'):
        self.data = _get_data(self.num_classes, 32)

    with tf.variable_scope('inputs'):
      self.x_input = tf.placeholder_with_default(
        self.data['x_batch'], [None, 32, 32, 3], 'X')
      self.y_input = tf.placeholder_with_default(
        self.data['y_batch'], [None, self.num_classes], 'Y')
      self.is_training = tf.placeholder(tf.bool, shape=None)

    with tf.variable_scope('preprocessing'):
      MEAN_IMAGE = tf.constant([0.5071, 0.4867, 0.4408], dtype=tf.float32)
      STD_IMAGE = tf.constant([0.2675, 0.2565, 0.2761], dtype=tf.float32)
      input_standardized = (self.x_input - MEAN_IMAGE) / STD_IMAGE
      input_standardized = tf.transpose(input_standardized, [0, 3, 1, 2])

      x = self._conv('init_conv', input_standardized, 3, 32, self._stride_arr(1))

    block_func = self._block
    filters = [32, 64, 128, 256, 512, 1024]
    '''''''''
    Because of the difference btw the input dimensions of CIFAR and ImageNet, 
    the stride of some blocks of MobileNet was changed to one.
    '''''''''
    with tf.variable_scope('unit_1_0'):
      x = block_func(x, filters[0], filters[1], 1, self.se1, self.se2)

    with tf.variable_scope('unit_2_0'):
      x = block_func(x, filters[1], filters[2], 2, self.se1, self.se2)
    for i in range(1, 2):
      with tf.variable_scope('unit_2_%d' % i):
        x = block_func(x, filters[2], filters[2], 1, self.se1, self.se2)

    with tf.variable_scope('unit_3_0'):
      x = block_func(x, filters[2], filters[3], 1, self.se1, self.se2)
    for i in range(1, 2):
      with tf.variable_scope('unit_3_%d' % i):
        x = block_func(x, filters[3], filters[3], 1, self.se1, self.se2)

    with tf.variable_scope('unit_4_0'):
      x = block_func(x, filters[3], filters[4], 2, self.se1, self.se2)
    for i in range(1, 6):
      with tf.variable_scope('unit_4_%d' % i):
        x = block_func(x, filters[4], filters[4], 1, self.se1, self.se2)

    with tf.variable_scope('unit_5_0'):
      x = block_func(x, filters[4], filters[5], 1, self.se1, self.se2)
    for i in range(1, 2):
      with tf.variable_scope('unit_5_%d' % i):
        x = block_func(x, filters[5], filters[5], 1, self.se1, self.se2)

    with tf.variable_scope('unit_last'):
      x = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      self.pre_softmax = self._fully_connected(x, self.num_classes)

    self.single_label = tf.cast(tf.argmax(self.y_input, axis=1), tf.int64)
    self.predictions = tf.argmax(self.pre_softmax, 1)
    self.correct_prediction = tf.equal(self.predictions, self.single_label)
    self.num_correct = tf.reduce_sum(
        tf.cast(self.correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(
        tf.cast(self.correct_prediction, tf.float32))

    with tf.variable_scope('costs'):
      self.y_xent = tf.nn.softmax_cross_entropy_with_logits(
          logits=self.pre_softmax, labels=self.y_input)
      self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
      self.mean_xent = tf.reduce_mean(self.y_xent)
      self.weight_decay_loss = self._decay()

  def _batch_norm(self, name, x, center=True, scale=True):
    """Batch normalization."""
    with tf.name_scope(name):
      return tf.layers.batch_normalization(
          inputs=x,
          momentum=.9,
          epsilon=1e-5,
          center=center,
          scale=scale,
          axis=1,
          training=self.is_training)

  def _block(self, x, in_filter, out_filter, stride, se1, se2):
    """Depthwise separable convolutions."""
    with tf.variable_scope('Depthwise'):
      x = self._depthwise_conv('depthwise_conv', x, 3, in_filter, stride)
      wh = 0
      if se1:
        wh = self._batch_norm('wh', x, False, False)
        wh = self._se_plus('se_plus', wh, in_filter, 4)
      x = self._batch_norm('bn', x) + wh
      x = self._relu(x)

    with tf.variable_scope('Pointwise'):
      x = self._conv('pointwise_conv', x, 1, out_filter, self._stride_arr(stride))
      wh = 0
      if se2:
        wh = self._batch_norm('wh', x, False, False)
        wh = self._se_plus('se_plus', wh, out_filter, 4)
      x = self._batch_norm('bn', x) + wh
      x = self._relu(x)

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  def _se_plus(self, name, x, filters, r):
      with tf.variable_scope(name):
        squeeze = self._squeeze(x)
        conv1 = self._conv('se_conv1', squeeze, 1, int(filters / r), self._stride_arr(1))
        ex_mid = tf.nn.relu(conv1)
        conv2 = self._conv('se_conv2', ex_mid, 1, filters, self._stride_arr(1))
        ex_scale = tf.sigmoid(conv2)
      return ex_scale * x + ex_scale

  def _squeeze(self, x):
      return self._avg_pool(x, (x.shape[2],x.shape[3]), 1)

  def _avg_pool(self, x, size, strides):
      return tf.layers.average_pooling2d(x, size, strides, 'valid', data_format='channels_first')

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if 'kernel' in var.name:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

  def _conv(self, name, x, filter_size, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      init = tf.random_normal_initializer(stddev=np.sqrt(2.0/n))
      layer = tf.layers.Conv2D(
          out_filters,
          kernel_size=filter_size,
          strides=strides,
          padding='same',
          data_format='channels_first',
          dilation_rate=(1,1),
          use_bias=False,
          kernel_initializer=init)
      return layer.apply(x)

  def _depthwise_conv(self, name, x, filter_size, in_channels, stride):
    with tf.variable_scope(name):
      n = filter_size * filter_size * in_channels
      init = tf.random_normal_initializer(stddev=np.sqrt(2.0/n))
      kernel = tf.get_variable(
          'kernel', [filter_size, filter_size, in_channels, 1],
          tf.float32, initializer=init)
      return tf.nn.depthwise_conv2d(x, kernel, (1, 1, stride, stride), 'SAME', data_format='NCHW')

  def _relu(self, x):
    """Relu."""
    return tf.nn.relu(x)

  def _batch_flatten(self, x):
      """
      Flatten the tensor except the first dimension.
      """
      shape = x.get_shape().as_list()[1:]
      if None not in shape:
          return tf.reshape(x, [-1, int(np.prod(shape))])
      return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))

  def _fully_connected(self, x, out_dim):
      """FullyConnected layer for final output."""
      inputs = self._batch_flatten(x)
      init = tf.uniform_unit_scaling_initializer(factor=1.0)
      layer = tf.layers.Dense(
          units=out_dim,
          use_bias=True,
          kernel_initializer=init,
          bias_initializer=tf.constant_initializer())
      return layer.apply(inputs)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [2, 3])
