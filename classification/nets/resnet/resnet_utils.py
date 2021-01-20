# -*- coding:utf-8 -*-
# Resnet网络的公用方法实现

import collections
import tensorflow as tf
from tensorflow.contrib import slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'stride'])):
    """A named tuple describing a ResNet block.
        Its parts are:
        scope: The scope of the `Block`.
        unit_fn: The ResNet unit function which takes as input a `Tensor` and
            returns another `Tensor` with the output of the ResNet unit.
        args: A list of length equal to the number of units in the `Block`. The list
            contains one (depth, depth_bottleneck, stride) tuple for each unit in the
            block to serve as argument to unit_fn.
    """


def subsample(inputs, factor, scope=None):
    """
    Subsamples input along the spatial dimensions.
    Args:
        inputs: A Tensor with shape [batch, height_in, width_in, channel_in]
        factor: An integer of subsampling factor

    Returns:
        output: A Tensor with shape [batch, height_out, width_out, channel_out]
    """

    if factor == 1:
        output = inputs
    else:
        output = slim.max_pool2d(
            inputs, [1, 1], factor, scope
        )
    return output


def conv2d_same(
        inputs,
        depth_out,
        kernel_size,
        stride,
        rate=1,
        scope=None
):
    """
    Args:
        inputs: A Tensor with shape [batch, height_in, width_in, channel_in],
        depth_out: An integer of channel number of output tensor
        kernel_size:
        stride:
        rate:
        scope:
    """

    if stride == 1:
        return slim.conv2d(
            inputs, num_outputs=depth_out, kernel_size=kernel_size, stride=1, rate=rate,
            padding='SAME', scope=scope
        )

    else:
        kernel_size_eff = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_eff - 1

        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        inputs = tf.pad(
            tensor=inputs,
            paddings=[[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
        )
        return slim.conv2d(
            inputs, num_outputs=depth_out, kernel_size=kernel_size, stride=stride, rate=rate,
            padding='VALID', scope=scope
        )
