# -*- coding:utf-8 -*-
# Resnet网络的公用方法实现

import collections
import tensorflow as tf
from tensorflow.contrib import slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
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
        output = slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)
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


class NoOpScope(object):
    """
    No Op context manager.
    """
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def stack_blocks_dense(
        net,
        blocks,
        output_stride=None,
        store_non_strided_activations=False,
        outputs_collections=None
):
    current_stride = 1
    rate = 1

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            block_stride = 1
            for i, unit in enumerate(block.args):

                if store_non_strided_activations and i == len(block.args) - 1:
                    block_stride = unit.get('stride', 1)
                    unit = dict(unit, stride=1)

                with tf.variable_scope('unit_{}'.format(i + 1), values=[net]):
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=1, **dict(unit, stride=1))
                        rate *= unit.get('rate', 1)
                    else:
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)

                        if output_stride is not None and current_stride > output_stride:
                            raise ValueError('The target output_stride cannot be reached.')

            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

            if output_stride is not None and current_stride == output_stride:
                rate *= block_stride

            else:
                net = subsample(net, block_stride)
                current_stride *= block_stride
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return net
