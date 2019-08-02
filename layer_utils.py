# coding: utf-8
#
# from __future__ import division, print_function
#
# import numpy as np
# import tensorflow as tf
#
# slim = tf.contrib.slim
#
#
# def conv2d(inputs, filters, kernel_size, strides=1):
#     def _fixed_padding(inputs, kernel_size):
#         pad_total = kernel_size - 1
#         pad_beg = pad_total // 2
#         pad_end = pad_total - pad_beg
#
#         padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
#                                         [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
#         return padded_inputs
#
#     if strides > 1:
#         inputs = _fixed_padding(inputs, kernel_size)
#     inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
#                          padding=('SAME' if strides == 1 else 'VALID'))
#     return inputs
#
#
# def darknet53_body(inputs):
#     def res_block(inputs, filters):
#         shortcut = inputs
#         net = conv2d(inputs, filters * 1, 1)
#         net = conv2d(net, filters * 2, 3)
#
#         net = net + shortcut
#
#         return net
#
#     # first two conv2d layers
#     net = conv2d(inputs, 32, 3, strides=1)
#     net = conv2d(net, 64, 3, strides=2)
#
#     # res_block * 1
#     net = res_block(net, 32)
#
#     net = conv2d(net, 128, 3, strides=2)
#
#     # res_block * 2
#     for i in range(2):
#         net = res_block(net, 64)
#
#     net = conv2d(net, 256, 3, strides=2)
#
#     # res_block * 8
#     for i in range(8):
#         net = res_block(net, 128)
#
#     route_1 = net
#     net = conv2d(net, 512, 3, strides=2)
#
#     # res_block * 8
#     for i in range(8):
#         net = res_block(net, 256)
#
#     route_2 = net
#     net = conv2d(net, 1024, 3, strides=2)
#
#     # res_block * 4
#     for i in range(4):
#         net = res_block(net, 512)
#     route_3 = net
#
#     return route_1, route_2, route_3
#
#
# def yolo_block(inputs, filters):
#     net = conv2d(inputs, filters * 1, 1)
#     net = conv2d(net, filters * 2, 3)
#     net = conv2d(net, filters * 1, 1)
#     net = conv2d(net, filters * 2, 3)
#     net = conv2d(net, filters * 1, 1)
#     route = net
#     net = conv2d(net, filters * 2, 3)
#     return route, net
#
#
# def upsample_layer(inputs, out_shape):
#     new_height, new_width = out_shape[1], out_shape[2]
#     # NOTE: here height is the first
#     # TODO: Do we need to set `align_corners` as True?
#     inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')
#     return inputs

# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.configs import cfgs
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
# import tfplot as tfp

def resnet_arg_scope(
        is_training=True, weight_decay=cfgs.WEIGHT_DECAY, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    '''

    In Default, we do not use BN to train resnet, since batch_size is too small.
    So is_training is False and trainable is False in the batch_norm params.

    '''
    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


# def add_heatmap(feature_maps, name):
#     '''
#
#     :param feature_maps:[B, H, W, C]
#     :return:
#     '''
#
#     def figure_attention(activation):
#         fig, ax = tfp.subplots()
#         im = ax.imshow(activation, cmap='jet')
#         fig.colorbar(im)
#         return fig
#
#     heatmap = tf.reduce_sum(feature_maps, axis=-1)
#     heatmap = tf.squeeze(heatmap, axis=0)
#     tfp.summary.plot(name, figure_attention, [heatmap])


def resnet_base(img_batch, scope_name, is_training=True):
    '''
    this code is derived from light-head rcnn.
    https://github.com/zengarden/light_head_rcnn

    It is convenient to freeze blocks. So we adapt this mode.
    '''
    if scope_name == 'resnet_v1_50':
        middle_num_units = 6
    elif scope_name == 'resnet_v1_101':
        middle_num_units = 23
    else:
        raise NotImplementedError('We only support resnet_v1_50 or resnet_v1_101. Check your network name....yjr')

    blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
              resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
              # use stride 1 for the last conv4 layer.

              resnet_v1_block('block3', base_depth=256, num_units=middle_num_units, stride=2),
              resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
              ]
              # when use fpn . stride list is [1, 2, 2]

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with tf.variable_scope(scope_name, scope_name):
            # Do the first few layers manually, because 'SAME' padding can behave inconsistently
            # for images of different sizes: sometimes 0, sometimes 1
            net = resnet_utils.conv2d_same(
                img_batch, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')

    not_freezed = [False] * cfgs.FIXED_BLOCKS + (4-cfgs.FIXED_BLOCKS)*[True]
    # Fixed_Blocks can be 1~3

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
        C2, _ = resnet_v1.resnet_v1(net,
                                    blocks[0:1],
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)

    # C2 = tf.Print(C2, [tf.shape(C2)], summarize=10, message='C2_shape')
    # add_heatmap(C2, 'Layer/C2')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
        C3, _ = resnet_v1.resnet_v1(C2,
                                    blocks[1:2],
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)
    # add_heatmap(C3, name='Layer/C3')
    # C3 = tf.Print(C3, [tf.shape(C3)], summarize=10, message='C3_shape')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[2]))):
        C4, _ = resnet_v1.resnet_v1(C3,
                                    blocks[2:3],
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)
    # add_heatmap(C4, name='Layer/C4')
    # C4 = tf.Print(C4, [tf.shape(C4)], summarize=10, message='C4_shape')
    return C4


def restnet_head(input, is_training, scope_name, stage):
    if stage == 'stage1':
        block4 = [resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            C5, _ = resnet_v1.resnet_v1(input,
                                        block4,
                                        global_pool=False,
                                        include_root_block=False,
                                        scope=scope_name)
            # C5 = tf.Print(C5, [tf.shape(C5)], summarize=10, message='C5_shape')
            flatten = tf.reduce_mean(C5, axis=[1, 2], keep_dims=False, name='global_average_pooling')
            # C5_flatten = tf.Print(C5_flatten, [tf.shape(C5_flatten)], summarize=10, message='C5_flatten_shape')

        # global average pooling C5 to obtain fc layers
    else:
        print('input shape is ', input.shape)
        try:
            fc_flatten = slim.flatten(input)
        except:
            fc_flatten = tf.reshape(input, [1, -1])
        net = slim.fully_connected(fc_flatten, 1024, scope='fc_1_{}'.format(stage))
        net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout_{}'.format(stage))
        flatten = slim.fully_connected(net, 1024, scope='fc_2_{}'.format(stage))
    return flatten




"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from .. import backend
from ..utils import anchors as utils_anchors

import numpy as np


class Anchors(keras.layers.Layer):
    """ Keras layer for generating achors for a given shape.
    """

    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        """ Initializer for an Anchors layer.

        Args
            size: The base size of the anchors to generate.
            stride: The stride of the anchors to generate.
            ratios: The ratios of the anchors to generate (defaults to AnchorParameters.default.ratios).
            scales: The scales of the anchors to generate (defaults to AnchorParameters.default.scales).
        """
        self.size   = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        if ratios is None:
            self.ratios  = utils_anchors.AnchorParameters.default.ratios
        elif isinstance(ratios, list):
            self.ratios  = np.array(ratios)
        if scales is None:
            self.scales  = utils_anchors.AnchorParameters.default.scales
        elif isinstance(scales, list):
            self.scales  = np.array(scales)

        self.num_anchors = len(ratios) * len(scales)
        self.anchors     = keras.backend.variable(utils_anchors.generate_anchors(
            base_size=size,
            ratios=ratios,
            scales=scales,
        ))

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = keras.backend.shape(features)

        # generate proposals from bbox deltas and shifted anchors
        if keras.backend.image_data_format() == 'channels_first':
            anchors = backend.shift(features_shape[2:4], self.stride, self.anchors)
        else:
            anchors = backend.shift(features_shape[1:3], self.stride, self.anchors)
        anchors = keras.backend.tile(keras.backend.expand_dims(anchors, axis=0), (features_shape[0], 1, 1))

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            if keras.backend.image_data_format() == 'channels_first':
                total = np.prod(input_shape[2:4]) * self.num_anchors
            else:
                total = np.prod(input_shape[1:3]) * self.num_anchors

            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

    def get_config(self):
        config = super(Anchors, self).get_config()
        config.update({
            'size'   : self.size,
            'stride' : self.stride,
            'ratios' : self.ratios.tolist(),
            'scales' : self.scales.tolist(),
        })

        return config


class UpsampleLike(keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        if keras.backend.image_data_format() == 'channels_first':
            source = backend.transpose(source, (0, 2, 3, 1))
            output = backend.resize_images(source, (target_shape[2], target_shape[3]), method='nearest')
            output = backend.transpose(output, (0, 3, 1, 2))
            return output
        else:
            return backend.resize_images(source, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        if keras.backend.image_data_format() == 'channels_first':
            return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


class RegressBoxes(keras.layers.Layer):
    """ Keras layer for applying regression values to boxes.
    """

    def __init__(self, mean=None, std=None, *args, **kwargs):
        """ Initializer for the RegressBoxes layer.

        Args
            mean: The mean value of the regression values which was used for normalization.
            std: The standard value of the regression values which was used for normalization.
        """
        if mean is None:
            mean = np.array([0, 0, 0, 0])
        if std is None:
            std = np.array([0.2, 0.2, 0.2, 0.2])

        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

        self.mean = mean
        self.std  = std
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return backend.bbox_transform_inv(anchors, regression, mean=self.mean, std=self.std)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        config.update({
            'mean': self.mean.tolist(),
            'std' : self.std.tolist(),
        })

        return config


class ClipBoxes(keras.layers.Layer):
    """ Keras layer to clip box values to lie inside a given shape.
    """

    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = keras.backend.cast(keras.backend.shape(image), keras.backend.floatx())
        if keras.backend.image_data_format() == 'channels_first':
            height = shape[2]
            width  = shape[3]
        else:
            height = shape[1]
            width  = shape[2]
        x1 = backend.clip_by_value(boxes[:, :, 0], 0, width)
        y1 = backend.clip_by_value(boxes[:, :, 1], 0, height)
        x2 = backend.clip_by_value(boxes[:, :, 2], 0, width)
        y2 = backend.clip_by_value(boxes[:, :, 3], 0, height)

        return keras.backend.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]





























