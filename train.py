# -*- coding:utf-8 -*-

from classification.model.resnet_v1_model import ResNetV1Model


if __name__ == '__main__':
    model = ResNetV1Model(
        model='resnet50',
        num_classes=1000
    )

    model.train(
        data_root_folder='/home/lifu/data/datasets/ILSVRC2012/train',
        checkpoint_folder_dir='/home/lifu/learning/Deep-Learning-with-Tensorflow/classification/ckpt',
        init_learning_rate=1e-4
    )