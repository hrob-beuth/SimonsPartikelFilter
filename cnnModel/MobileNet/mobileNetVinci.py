from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

BASE_WEIGHT_PATH = ('https://storage.googleapis.com/tensorflow/'
                    'keras-applications/mobilenet_v2/')
layers = None


@keras_export('keras.applications.mobilenet_v2.MobileNetV2',
              'keras.applications.MobileNetV2')
def MobileNetVinci(input_shape=None,
                alpha=1.0,
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                pooling=None,
                classes=1000,
                classifier_activation='softmax',
                **kwargs):
  
  global layers
  layers = VersionAwareLayers()
  default_size = 224
  rows = input_shape[0]
  cols = input_shape[1]
  
  input_shape = imagenet_utils.obtain_input_shape(
      input_shape,
      default_size=default_size,
      min_size=32,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  img_input = layers.Input(shape=input_shape)
  channel_axis = -1

  first_block_filters = _make_divisible(32 * alpha, 8)

  x = tf.pad(img_input, [[0,0],[1,1],[1,1],[0,0]])
  x = layers.Conv2D(
      first_block_filters,
      kernel_size=3,
      strides=(2, 2),
      padding='valid',
      use_bias=False,
      name='Conv1')(
          x)
  x = layers.BatchNormalization(
      axis=channel_axis, epsilon=1e-3, momentum=0.999, name='bn_Conv1')(
          x)
  x = layers.ReLU(6., name='Conv1_relu')(x)

  x = _inverted_res_block(
      x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

  x = _inverted_res_block(
      x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
  x = _inverted_res_block(
      x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

  x = _inverted_res_block(
      x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
  x = _inverted_res_block(
      x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
  x = _inverted_res_block(
      x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

  x = _inverted_res_block(
      x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
  x = _inverted_res_block(
      x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
  x = _inverted_res_block(
      x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
  x = _inverted_res_block(
      x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)

  x = _inverted_res_block(
      x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
  x = _inverted_res_block(
      x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
  x = _inverted_res_block(
      x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)


  if alpha > 1.0:
    last_block_filters = _make_divisible(1280 * alpha, 8)
  else:
    last_block_filters = 1280

  x = layers.Conv2D(
      last_block_filters, kernel_size=1, use_bias=False, name='Conv_1')(
          x)
  x = layers.BatchNormalization(
      axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(
          x)
  x = layers.ReLU(6., name='out_relu')(x)

  if include_top:
    x = layers.GlobalAveragePooling2D()(x)
    imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Dense(classes, activation=classifier_activation,
                     name='predictions')(x)

  # Create model.
  model = training.Model(inputs, x, name='mobilenetv2_%0.2f_%s' % (alpha, rows))

  return model


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
  """Inverted ResNet block."""
  channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

  in_channels = backend.int_shape(inputs)[channel_axis]
  pointwise_conv_filters = int(filters * alpha)
  pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
  x = inputs
  prefix = 'block_{}_'.format(block_id)

  if block_id:
    # Expand
    x = layers.Conv2D(
        expansion * in_channels,
        kernel_size=1,
        padding='same',
        use_bias=False,
        activation=None,
        name=prefix + 'expand')(
            x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'expand_BN')(
            x)
    x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
  else:
    prefix = 'expanded_conv_'

  # Depthwise
  if stride == 2:
    x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]])
  x = layers.DepthwiseConv2D(
      kernel_size=3,
      strides=stride,
      activation=None,
      use_bias=False,
      padding='same' if stride == 1 else 'valid',
      name=prefix + 'depthwise')(
          x)
  x = layers.BatchNormalization(
      axis=channel_axis,
      epsilon=1e-3,
      momentum=0.999,
      name=prefix + 'depthwise_BN')(
          x)

  x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

  # Project
  x = layers.Conv2D(
      pointwise_filters,
      kernel_size=1,
      padding='same',
      use_bias=False,
      activation=None,
      name=prefix + 'project')(
          x)
  x = layers.BatchNormalization(
      axis=channel_axis,
      epsilon=1e-3,
      momentum=0.999,
      name=prefix + 'project_BN')(
          x)

  if in_channels == pointwise_filters and stride == 1:
    return layers.Add(name=prefix + 'add')([inputs, x])
  return x


def _make_divisible(v, divisor, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v


@keras_export('keras.applications.mobilenet_v2.preprocess_input')
def preprocess_input(x, data_format=None):
  return imagenet_utils.preprocess_input(x, data_format=data_format, mode='tf')


@keras_export('keras.applications.mobilenet_v2.decode_predictions')
def decode_predictions(preds, top=5):
  return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode='',
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__
