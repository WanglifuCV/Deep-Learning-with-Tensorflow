# -*- coding:utf-8 -*-
from tensorflow.contrib import slim
import tensorflow as tf
from xnncloud_classification.nets.resnet import resnet_utils


@slim.add_arg_scope
def bottleneck(
        inputs,
        depth,
        depth_bottleneck,
        stride,
        rate=1,
        scope=None,
        use_bound_activation=False,
        outputs_collections=None
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

    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
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
            1,
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

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                output)


@slim.add_arg_scope
def resnet_v1(
        inputs,
        blocks,
        num_class=None,
        is_training=True,
        global_pool=True,
        output_stride=None,
        include_root_block=True,
        spatial_squeeze=True,
        store_non_strided_activations=False,
        reuse=None,
        scope=None
):
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        # 不知道有何用，需要看一下效果
        end_points_collection = sc.original_name_scope + '_end_points'
        print(end_points_collection)

        with slim.arg_scope(
            [slim.conv2d, bottleneck],
            outputs_collections=end_points_collection
        ):
            with (slim.arg_scope(
                [slim.batch_norm], is_training=is_training
            )) if is_training else resnet_utils.NoOpScope:

                net = inputs

                if include_root_block:
                    net = resnet_utils.conv2d_same(
                        net,
                        64,
                        7,
                        stride=2,
                        scope='conv1'
                    )
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

                net = resnet_utils.stack_blocks_dense(
                    net,
                    blocks,
                    output_stride,
                    store_non_strided_activations=store_non_strided_activations
                )

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                if global_pool:
                    # Global Average Pooling
                    net = tf.reduce_mean(
                        input_tensor=net, axis=[1, 2], name='pool5', keepdims=True
                    )

                if num_class:
                    net = slim.conv2d(
                        net,
                        num_class,
                        [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='logits'
                    )
                    end_points[sc.name + '/logits'] = net

                if spatial_squeeze:
                    net = tf.squeeze(net, axis=[1, 2], name='spatial_squeeze')
                    end_points[sc.name + '/spatial_squeeze'] = net

                end_points['predictions'] = slim.softmax(net, scope='predictions')

            return net, end_points


def resnet_v1_block(scope, base_depth, num_units, stride):

    return resnet_utils.Block(
        scope,
        bottleneck,
        [{
            'depth': base_depth * 4,
            'depth_bottleneck': base_depth,
            'stride': 1
        }] * (num_units - 1) + [{
            'depth': base_depth * 4,
            'depth_bottleneck': base_depth,
            'stride': stride
        }]
    )


def resnet_v1_50(
        inputs,
        num_classes=None,
        is_training=True,
        global_pool=True,
        output_stride=None,
        spatial_squeeze=True,
        store_non_strided_activations=False,
        min_base_depth=8,
        depth_multiplier=1,
        reuse=None,
        scope='resnet_v1_50'
):
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)

    blocks = [
        resnet_v1_block('block1', base_depth=depth_func(64), num_units=3,
                        stride=2),
        resnet_v1_block('block2', base_depth=depth_func(128), num_units=4,
                        stride=2),
        resnet_v1_block('block3', base_depth=depth_func(256), num_units=6,
                        stride=2),
        resnet_v1_block('block4', base_depth=depth_func(512), num_units=3,
                        stride=1),
    ]
    return resnet_v1(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     store_non_strided_activations=store_non_strided_activations,
                     reuse=reuse, scope=scope)


if __name__ == '__main__':
    graph = tf.get_default_graph()
    with graph.as_default():

        inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], 'input/placeholder')

        net = resnet_v1_50(
            inputs,
            num_classes=1000,
            is_training=True,
            global_pool=True
        )

        tf.summary.FileWriter('log', graph)
