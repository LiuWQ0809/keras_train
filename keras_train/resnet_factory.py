# ResNet implementation
# ReNet50 code copied from:
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py
# Modified into squeeze-and-excitation network by Jin Huang
# Date: 09/26

# ResNet152 implemented by Jin Huang refering ResNet50
# Modified into squeeze-and-excitation network by Jin Huang
# Date: 09/27

# ResNet builder: copied from https://github.com/raghakot/keras-resnet/blob/master/resnet.py

import numpy as np
import warnings
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D, Conv2DTranspose
from keras.layers import Input, Dropout, Dense, Flatten, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras import regularizers
from keras import initializers
from keras.models import Model
from keras import backend
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.layers import *
import os
import warnings
from keras import layers
from keras.models import Model
from keras_applications import *  # import get_submodules_from_kwargs
from keras_applications.imagenet_utils import _obtain_input_shape
import pdb

# from tensorflow.keras.layers import MaxPooling2D, Convolution2D, AveragePooling2D
# from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, Activation, Reshape
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import concatenate
# from tensorflow.keras import regularizers
# from tensorflow.keras import initializers
# from tensorflow.keras.models import Model
# from tensorflow.keras import backend
# from tensorflow.keras.utils import convert_all_kernels_in_model
# from tensorflow.keras.utils import get_file
# from tensorflow.keras.layers import *
# from tensorflow.keras import layers
# from tensorflow.keras.models import Model




WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')

WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

ratio = 16

##########################################################
            #For ResNet 18 and 34#
##########################################################

def identity_block_18(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 3*3
    x = layers.Conv2D(filters1, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    # 3*3
    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block_18(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, kernel_size, strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    shortcut = layers.Conv2D(filters2, kernel_size, strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def ResNet18_base(input):
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block_18(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block_18(x, 3, [64, 64], stage=2, block='b')

    x = conv_block_18(x, 3, [128, 128], stage=3, block='a')
    x = identity_block_18(x, 3, [128, 128], stage=3, block='b')

    x = conv_block_18(x, 3, [256, 256], stage=4, block='a')
    x = identity_block_18(x, 3, [256, 256], stage=4, block='b')

    x = conv_block_18(x, 3, [512, 512], stage=5, block='a')
    x = identity_block_18(x, 3, [512, 512], stage=5, block='b')

    return x

def ResNet34_base(input):
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block_18(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block_18(x, 3, [64, 64], stage=2, block='b')
    x = identity_block_18(x, 3, [64, 64], stage=2, block='c')

    x = conv_block_18(x, 3, [128, 128], stage=3, block='a')
    x = identity_block_18(x, 3, [128, 128], stage=3, block='b')
    x = identity_block_18(x, 3, [128, 128], stage=3, block='c')
    x = identity_block_18(x, 3, [128, 128], stage=3, block='d')

    x = conv_block_18(x, 3, [256, 256], stage=4, block='a')
    x = identity_block_18(x, 3, [256, 256], stage=4, block='b')
    x = identity_block_18(x, 3, [256, 256], stage=4, block='c')
    x = identity_block_18(x, 3, [256, 256], stage=4, block='d')
    x = identity_block_18(x, 3, [256, 256], stage=4, block='e')
    x = identity_block_18(x, 3, [256, 256], stage=4, block='f')

    x = conv_block_18(x, 3, [512, 512], stage=5, block='a')
    x = identity_block_18(x, 3, [512, 512], stage=5, block='b')
    x = identity_block_18(x, 3, [512, 512], stage=5, block='c')

    return x



##########################################################
            #For ResNet 50, 101, 152#
##########################################################
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def se_layer(input_x, out_dim, ratio):
    # Squeeze
    squeeze = GlobalAveragePooling2D()(input_x)

    # Excitation
    excitation = Dense(units = int(out_dim/ratio), activation="relu")(squeeze)
    excitation = Dense(units = out_dim, activation = "sigmoid")(excitation)
    excitation = Reshape((1, 1, out_dim))(excitation)

    # Scale
    scale = multiply([input_x, excitation])

    return scale

def se_identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)

    # Add SE block
    nb_channel = int(x.get_shape()[-1])
    x = se_layer(input_x=x, out_dim=nb_channel,ratio=ratio)

    return x

def se_conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    # Add SE block
    nb_channel = int(x.get_shape()[-1])
    x = se_layer(input_x=x, out_dim=nb_channel, ratio=ratio)

    return x

def se_ResNet50(include_top=True, weights='imagenet', input_tensor=None,
                pooling=None, classes=2):

    if backend.image_data_format() == 'channels_first':
        img_input = Input((3, 224, 224))
        bn_axis = 1
    else:
        img_input = Input((224, 224, 3))
        bn_axis = -1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = se_conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = se_identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = se_identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = se_conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = se_identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = se_identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = se_identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = se_conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = se_identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = se_identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = se_identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = se_identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = se_identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = se_conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = se_identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = se_identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')
    # print model.summary()

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path, by_name=True)

    elif weights is not None:
        model.load_weights(weights)

    return model

# def ResNet50(include_top=True,
#              weights='imagenet',
#              pretrain=True,
#              input_tensor=None,
#              input_shape=None,
#              pooling=None,
#              classes=1000,
#              **kwargs):
#     """Instantiates the ResNet50 architecture.
#     Optionally loads weights pre-trained on ImageNet.
#     Note that the data format convention used by the model is
#     the one specified in your Keras config at `~/.keras/keras.json`.
#     # Arguments
#         include_top: whether to include the fully-connected
#             layer at the top of the network.
#         weights: one of `None` (random initialization),
#               'imagenet' (pre-training on ImageNet),
#               or the path to the weights file to be loaded.
#         input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
#             to use as image input for the model.
#         input_shape: optional shape tuple, only to be specified
#             if `include_top` is False (otherwise the input shape
#             has to be `(224, 224, 3)` (with `channels_last` data format)
#             or `(3, 224, 224)` (with `channels_first` data format).
#             It should have exactly 3 inputs channels,
#             and width and height should be no smaller than 197.
#             E.g. `(200, 200, 3)` would be one valid value.
#         pooling: Optional pooling mode for feature extraction
#             when `include_top` is `False`.
#             - `None` means that the output of the model will be
#                 the 4D tensor output of the
#                 last convolutional layer.
#             - `avg` means that global average pooling
#                 will be applied to the output of the
#                 last convolutional layer, and thus
#                 the output of the model will be a 2D tensor.
#             - `max` means that global max pooling will
#                 be applied.
#         classes: optional number of classes to classify images
#             into, only to be specified if `include_top` is True, and
#             if no `weights` argument is specified.
#     # Returns
#         A Keras model instance.
#     # Raises
#         ValueError: in case of invalid argument for `weights`,
#             or invalid input shape.
#     """
#     global backend, layers, models, keras_utils
#     backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

#     if not (weights in {'imagenet', None} or os.path.exists(weights)):
#         raise ValueError('The `weights` argument should be either '
#                          '`None` (random initialization), `imagenet` '
#                          '(pre-training on ImageNet), '
#                          'or the path to the weights file to be loaded.')

#     if weights == 'imagenet' and include_top and classes != 1000:
#         raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
#                          ' as true, `classes` should be 1000')

#     # Determine proper input shape
#     input_shape = _obtain_input_shape(input_shape,
#                                       default_size=224,
#                                       min_size=32,
#                                       data_format=backend.image_data_format(),
#                                       require_flatten=include_top,
#                                       weights=weights)

#     if input_tensor is None:
#         img_input = layers.Input(shape=input_shape)
#     else:
#         if not backend.is_keras_tensor(input_tensor):
#             img_input = layers.Input(tensor=input_tensor, shape=input_shape)
#         else:
#             img_input = input_tensor
#     if backend.image_data_format() == 'channels_last':
#         bn_axis = 3
#     else:
#         bn_axis = 1

#     x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
#     x = layers.Conv2D(64, (7, 7),
#                       strides=(2, 2),
#                       padding='valid',
#                       kernel_initializer='he_normal',
#                       name='conv1')(x)
#     x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
#     x = layers.Activation('relu')(x)
#     x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
#     x1 = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

#     x = conv_block(x1, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
#     x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
#     x2 = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

#     x = conv_block(x2, 3, [128, 128, 512], stage=3, block='a')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
#     x3 = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

#     x = conv_block(x3, 3, [256, 256, 1024], stage=4, block='a')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
#     x4 = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

#     x = conv_block(x4, 3, [512, 512, 2048], stage=5, block='a')
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
#     x5 = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
#     pdb.set_trace()
#     if include_top:
#         x = layers.GlobalAveragePooling2D(name='avg_pool')(x5)
#         x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
#     else:
#         if pooling == 'avg':
#             x = layers.GlobalAveragePooling2D()(x5)
#         elif pooling == 'max':
#             x = layers.GlobalMaxPooling2D()(x5)
#         else:
#             warnings.warn('The output shape of `ResNet50(include_top=False)` '
#                           'has been changed since Keras 2.2.0.')

#     # Ensure that the model takes into account
#     # any potential predecessors of `input_tensor`.
#     if input_tensor is not None:
#         inputs = keras_utils.get_source_inputs(input_tensor)
#     else:
#         inputs = img_input
#     # Create model.
#     model = Model(inputs, x, name='resnet50')
#     print(model.Summary())

#     # Load weights.
#     if weights == 'imagenet':
#         if include_top:
#             weights_path = keras_utils.get_file(
#                 'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
#                 WEIGHTS_PATH,
#                 cache_subdir='models',
#                 md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
#         else:
#             weights_path = keras_utils.get_file(
#                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
#                 WEIGHTS_PATH_NO_TOP,
#                 cache_subdir='models',
#                 md5_hash='a268eb855778b3df3c7506639542a6af')
#         model.load_weights(weights_path)
#         if backend.backend() == 'theano':
#             keras_utils.convert_all_kernels_in_model(model)
#     elif weights is not None:
#         model.load_weights(weights)

#     return model, x5

def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, models, keras_utils
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')
    # pdb.set_trace()
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x1 = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x1, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x2 = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x2, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x3 = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x3, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x4 = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    # pdb.set_trace()
    x = conv_block(x4, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x5 = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x6 = Conv2DTranspose(1024, 2, strides=(2, 2))(x5)
    x4_6 = layers.add([x4, x6])
    x = layers.Conv2D(2048, 1, strides=(1, 1),
                      kernel_initializer='he_normal',
                      name='conv4_add_5')(x4_6)
    x = layers.Activation('relu')(x)
    # x7 =
    # x8 =
    # x9 = 

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        print("*"*40)
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')
    # print(model.summary())

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model

def ResNet101(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='0', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='1')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='2')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='0')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='1')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='2')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='3')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='0')
    for i in range(1, 37):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='0')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='1')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='2')

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')
    print(model.Summary())

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model

def ResNet152(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='0', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='1')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='2')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='0')
    for i in range(1, 9):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block=str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='0')
    for j in range(1, 37):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=str(j))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')
    print(model.Summary())

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return mode

# Code initially from:
# https://github.com/kurapan/EAST/blob/master/model.py
# Modified by: Jin Huang

# from keras.applications.resnet50 import ResNet50
# from keras.models import Model
# from keras.layers import Conv2D, concatenate, BatchNormalization, Lambda, Input, multiply, add, ZeroPadding2D, Activation, Layer, MaxPooling2D, Dropout
# from keras import regularizers
# import keras.backend as K
# import tensorflow as tf
# import numpy as np

# from keras.applications.resnet50 import ResNet50
# from keras.models import Model
# from keras.layers import Conv2D, concatenate, BatchNormalization, Lambda, Input, multiply, add, ZeroPadding2D, Activation, Layer, MaxPooling2D, Dropout
# from keras import regularizers
# import keras.backend.tensorflow_backend as K

# # from tensorflow.keras.applications.resnet50 import ResNet50
# # from tensorflow.keras.models import Model
# # from tensorflow.keras.layers import Conv2D, concatenate, BatchNormalization, Lambda, Input, multiply, add, ZeroPadding2D, Activation, Layer, MaxPooling2D, Dropout
# # from tensorflow.keras import regularizers
# # from tensorflow.python.keras import backend as K

# RESIZE_FACTOR = 2

# def resize_bilinear(x):
#     return tf.image.resize_bilinear(x, size=[K.shape(x)[1]*RESIZE_FACTOR, K.shape(x)[2]*RESIZE_FACTOR])

# def resize_output_shape(input_shape):
#     shape = list(input_shape)
#     assert len(shape) == 4
#     shape[1] *= RESIZE_FACTOR
#     shape[2] *= RESIZE_FACTOR

#     return tuple(shape)

# def east_resnet50(input_width, input_height,):
#     if K.image_data_format() == 'channels_first':
#         channel_axis = 1
#         inputs = Input((3, input_width, input_height))
#     else:
#         channel_axis = -1
#         inputs = Input((input_width, input_height, 3))

#     resnet = ResNet50(input_tensor=inputs,
#                         weights='imagenet',
#                         include_top=False,
#                         pooling=None)
#     x = resnet.get_layer('activation_49').output

#     x = Lambda(resize_bilinear, name='resize_1')(x)
#     x = concatenate([x, resnet.get_layer('activation_40').output], axis=channel_axis)
#     x = Conv2D(128, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
#     x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
#     x = Activation('relu')(x)
#     x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
#     x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
#     x = Activation('relu')(x)

#     x = Lambda(resize_bilinear, name='resize_2')(x)
#     x = concatenate([x, resnet.get_layer('activation_22').output], axis=channel_axis)
#     x = Conv2D(64, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
#     x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
#     x = Activation('relu')(x)
#     x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
#     x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
#     x = Activation('relu')(x)

#     x = Lambda(resize_bilinear, name='resize_3')(x)
#     x = concatenate([x, ZeroPadding2D(((1, 0), (1, 0)))(resnet.get_layer('activation_10').output)], axis=channel_axis)
#     x = Conv2D(32, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
#     x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
#     x = Activation('relu')(x)
#     x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
#     x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
#     x = Activation('relu')(x)

#     x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
#     x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
#     x = Activation('relu')(x)