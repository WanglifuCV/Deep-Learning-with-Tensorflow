# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/wanglifu/learning/Deep-Learning-with-TensorFlow/Deep-Learning-with-Tensorflow')

from tensorflow.contrib import slim
import tensorflow as tf
from classification.nets.resnet import resnet_utils


def bottleneck(
        inputs,
        depth,
        depth_bottleneck,
        stride,
        rate=1,
        scope=None,
        use_bound_activation=False
):
    """
    Bottleneck block
    Args:
        inputs: A Tensor with shape [batch, height_in, width_in, channel_in]
        depth: An integer of channel number of bottleneck output Tensor
        depth_bottleneck: An integer of channel number of bottleneck layer
        stride: The Bottleneck unit stride.
        rate: An integer, rate for atrous convolution.
        scope: None
        use_bound_activation: False
    Returns:
        The ResNet bottleneck output tensor

    """

    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]):
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)

        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, factor=stride, scope='shortcut')
        else:
            shortcut = slim.conv2d(
                inputs,
                depth,
                [1, 1],
                stride,
                activation_fn=None,
                scope='shortcut'
            )

        residual = slim.conv2d(
            inputs,
            depth_bottleneck,
            [1, 1],
            stride,
            scope='conv1'
        )

        residual = resnet_utils.conv2d_same(
            inputs=residual,
            depth_out=depth_bottleneck,
            kernel_size=3,
            stride=stride,
            rate=rate,
            scope='conv2'
        )
        residual = slim.conv2d(
            residual,
            depth,
            [1, 1],
            stride=1,
            activation_fn=None,
            scope='conv3'
        )

        if use_bound_activation:
            residual = tf.clip_by_norm(residual, -6.0, 6.0)
            output = tf.nn.relu6(shortcut + residual)
        else:
            output = tf.nn.relu(shortcut + residual)

        return output


if __name__ == '__main__':
    graph = tf.get_default_graph()
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, [None, 56, 56, 64], 'input/placeholder')

        bottleneck_output = bottleneck(
            inputs=inputs, 
            depth=64, 
            depth_bottleneck=32,
            stride=1,
            scope='test-bottleneck'
        )

        train_writer = tf.summary.FileWriter('log', graph)