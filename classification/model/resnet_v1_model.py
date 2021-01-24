# -*- coding:utf-8 -*-

from operator import mod
import tensorflow as tf
from classification.nets.resnet import resnet_v1
from classification.data.generator1 import DataLoader
import os.path as osp


class ResNetV1Model(object):

    def __init__(
        self,
        model: str,
        num_classes: int
    ) -> None:
        self.model = model
        self.num_classes = num_classes

    def build_model(
            self,
            inputs,
            is_training,
            global_pool=True,
            scope='resnet_v1_50'
        ):
        if self.model.lower() == 'resnet50':
            model_fn = resnet_v1.resnet_v1_50
        else:
            raise ValueError('Model type error: {} given'.format(self.model.lower()))
        logits, end_points = model_fn(
            inputs,
            num_classes=self.num_classes,
            is_training=is_training,
            global_pool=global_pool,
            output_stride=None,
            spatial_squeeze=True,
            store_non_strided_activations=False,
            min_base_depth=8,
            depth_multiplier=1,
            reuse=None,
            scope=scope
        )

        return logits, end_points

    def train(
        self,
        data_root_folder,
        checkpoint_folder_dir,
        batch_size=64,
        epoch_num=90,
        init_learning_rate=1e-4,
        pretrain_model_dir=None
    ):

        # 获取数据
        loader = DataLoader(data_root_folder, class_num=self.num_classes)

        images_batch, onehot_labels_batch = loader.get_batch(batch_size=batch_size)
        # 模型搭建
        logits, _ = self.build_model(
            inputs=images_batch,
            is_training=True
        )

        global_step = tf.train.get_or_create_global_step()

        # 计算loss
        tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels_batch, logits=logits)
        total_loss = tf.losses.get_total_loss()

        learning_rate = tf.train.exponential_decay(
            learning_rate=init_learning_rate,
            global_step=global_step,
            decay_steps=1e5, decay_rate=0.5
        )

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step)

        top_1_accuracy, top_5_accuracy = self.build_accuracy_base(logits=logits, onehot_labels_batch=onehot_labels_batch)

        tf.summary.scalar('top_1_acc', top_1_accuracy)
        tf.summary.scalar('top_5_acc', top_5_accuracy)
        tf.summary.scalar('total_loss', total_loss)

        merged_summary = tf.summary.merge_all()

        saver = tf.train.Saver()

        with tf.Session() as sess:

            train_summary_writer = tf.summary.FileWriter(osp.join(checkpoint_folder_dir, 'log'), sess.graph)
            init = tf.global_variables_initializer()

            sess.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                for epoch in range(epoch_num * 1000):
                    if coord.should_stop():
                        raise RuntimeError('Data load error!')
                        # print('Done!')
                        break
                    print('Step {}'.format(epoch))
                    [loss, _, train_summary, step, top_1_acc, top_5_acc] = sess.run([
                        total_loss, train_op, merged_summary, global_step,
                        top_1_accuracy, top_5_accuracy
                    ])

                    print('Loss: {}'.format(loss))
                    print('Top 1 Accuracy: {}'.format(top_1_acc))
                    print('Top 5 Accuracy: {}'.format(top_5_acc))

                    train_summary_writer.add_summary(train_summary, step)

                    if epoch % 1000 == 0:
                        saver.save(sess, osp.join(checkpoint_folder_dir, 'resnet50'), global_step=step)


            except tf.errors.OutOfRangeError:
                print('Epoch {} finished.'.format(epoch))

            finally:
                coord.request_stop()
            coord.join(threads)

    def build_accuracy_base(self, logits, onehot_labels_batch):
        # 计算accuracy
        labels_batch = tf.cast(tf.argmax(onehot_labels_batch, axis=1), tf.int32)
        predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        correct_predictions = tf.equal(predictions, labels_batch)
        # top_1_accuracy = tf.metrics.accuracy(tf.argmax(labels_batch, axis=1), predictions, name='acc')

        top_1_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        top_5_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=logits, targets=labels_batch, k=5), tf.float32))

        return top_1_accuracy, top_5_accuracy

