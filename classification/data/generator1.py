# -*- coding:utf-8 -*-
import numpy as np
from sklearn.utils import shuffle
import os
import os.path as osp
import tensorflow as tf


def GenerateData(epoch_num, batch_size=100):
    """
    随机生成x, y的数据
    """
    for i in range(epoch_num):
        train_x = np.linspace(-1, 1, batch_size)
        train_y = np.random.randn(*train_x.shape)

        yield i, shuffle(train_x, train_y)


def load_image_sample(sample_root_dir):
    """
    读取根目录sample_root_dir下所有文件夹的全部文件，同时以文件夹名称为label
    
    """
    image_path_list = []
    lable_name_list = []
    for (dir_path, dir_names, file_names) in os.walk(sample_root_dir):
        for file_name in file_names:
            file_name_path = os.sep.join([dir_path, file_name])
            image_path_list.append(file_name_path)
            lable_name_list.append(osp.basename(dir_path))

    lab = list(sorted(set(lable_name_list)))
    lab_dict = dict(zip(lab, list(range(len(lab)))))

    labels = [lab_dict[i] for i in lable_name_list]

    return shuffle(np.asarray(image_path_list), np.asarray(labels)), np.asarray(lab)


def get_batches(image, label, resize_w, resize_h, channels, batch_size, class_num):

    queue = tf.train.slice_input_producer([image, label])

    label = queue[1]

    image_c = tf.read_file(queue[0])

    image = tf.image.decode_jpeg(image_c, channels)

    image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)

    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64)

    images_batch = tf.cast(image_batch, tf.float32)

    labels_batch = tf.reshape(label_batch, [batch_size])

    if class_num:
        labels_batch = tf.one_hot(labels_batch, class_num)

    return images_batch, labels_batch


class DataLoader(object):

    def __init__(
            self,
            data_root_dir: str,
            class_num: int
    ) -> None:
        self.data_root_dir = data_root_dir
        (self.images_file_list, self.images_label_list), self.category_names = load_image_sample(
            data_root_dir
        )
        if len(self.category_names) <= class_num:
            # 若统计得到的label_name比class_num要少，说明class_num设大了，把class_num设为len(self.label_name)
            self.class_num = len(self.category_names)
            self.train_categories = self.category_names
        else:
            # 若需要
            self.class_num = class_num
            self.train_categories = np.random.choice(self.category_names, class_num)

    def filte_sample_category(self):
        images = []
        labels = []
        for image, label in zip(self.images_file_list, self.images_label_list):
            if osp.basename(osp.dirname(image)) not in self.train_categories:
                continue
            else:
                images.append(image)
                labels.append(label)

        return np.asarray(images), np.asarray(labels)

    def get_batch(
            self,
            batch_size: int = 64,
            height: int = 224,
            width: int = 224
    ):
        images_file_list, images_label_list = self.filte_sample_category()
        image_batches, label_batches = get_batches(
            images_file_list,
            images_label_list,
            height, width, 3, batch_size,
            len(self.train_categories)
        )

        return image_batches, label_batches


if __name__ == '__main__':

    loader = DataLoader(
        data_root_dir='/home/lifu/data/datasets/ILSVRC2012/train',
        class_num=300
    )

    images_tensor, labels_tensor = loader.get_batch(
        batch_size=64,
        height=224,
        width=224
    )

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run([init])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for step in np.arange(10):
                if coord.should_stop():
                    break

                images, labels = sess.run([images_tensor, labels_tensor])

                print(images)
                print(labels)
        except tf.errors.OutOfRangeError:
            print('Done!')

        finally:
            coord.request_stop()

        coord.join(threads)
