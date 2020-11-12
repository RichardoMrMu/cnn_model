# -*- coding: utf-8 -*-
# @Time    : 2020-11-09 21:46
# @Author  : RichardoMu
# @File    : resnet.py
# @Software: PyCharm
import tensorflow as tf
from resnet_block import _make_basic_block_layer,_make_bottleneck_layer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class ResNetBlk1(tf.keras.Model):
    def __init__(self,layers_param,num_class):
        super(ResNetBlk1, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,kernel_size=7,padding='same',strides=2)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same')
        self.layer1 = _make_basic_block_layer(filter_num=64,blocks=layers_param[0])
        self.layer2 = _make_basic_block_layer(filter_num=128, blocks=layers_param[1],strides=2)
        self.layer3 = _make_basic_block_layer(filter_num=256, blocks=layers_param[2],strides=2)
        self.layer4 = _make_basic_block_layer(filter_num=512, blocks=layers_param[3],strides=2)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(1000)
        self.dense2 = tf.keras.layers.Dense(num_class,activation='softmax')

    def call(self,x,training=None):
        output = self.conv1(x)
        output = self.bn1(output,training=training)
        output = self.max_pool(output)
        output = self.layer1(output,training=training)
        output = self.layer2(output, training=training)
        output = self.layer3(output, training=training)
        output = self.layer4(output, training=training)
        output = self.avg_pool(output)
        output = self.dense1(output)
        output = self.dense2(output)
        return output


class ResNetBlk2(tf.keras.Model):
    def __init__(self,layers_param,num_class):
        super(ResNetBlk2, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, padding='same', strides=2)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')
        self.layer1 = _make_bottleneck_layer(filter_num=64, blocks=layers_param[0])
        self.layer2 = _make_bottleneck_layer(filter_num=128, blocks=layers_param[1],strides=2)
        self.layer3 = _make_bottleneck_layer(filter_num=256, blocks=layers_param[2],strides=2)
        self.layer4 = _make_bottleneck_layer(filter_num=512, blocks=layers_param[3],strides=2)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(1000)
        self.dense2 = tf.keras.layers.Dense(num_class)

    def call(self, x, training=None):
        output = self.conv1(x)
        output = self.bn1(output, training=training)
        output = self.max_pool(output)
        output = self.layer1(output, training=training)
        output = self.layer2(output, training=training)
        output = self.layer3(output, training=training)
        output = self.layer4(output, training=training)
        print(output.shape)
        output = self.avg_pool(output)
        print(output.shape)
        output = self.dense1(output)
        output = self.dense2(output)
        return output


def resnet_18(num_class=10):
    return ResNetBlk1(layers_param=[2,2,2,2],num_class=num_class)


def resnet_34(num_class=10):
    return ResNetBlk1(layers_param=[3,4,6,3],num_class=num_class)


def resnet_50(num_class=10):
    return ResNetBlk2(layers_param=[3,4,6,3],num_class=num_class)


def resnet_101(num_class=10):
    return ResNetBlk2(layers_param=[3,4,23,3],num_class=num_class)


def resnet_152(num_class=10):
    return ResNetBlk2(layers_param=[3,8,36,3],num_class=num_class)


def main():
    # resnet18 = resnet_18()
    # resnet34 = resnet_34()
    resnet50 = resnet_50()
    # resnet101 = resnet_101()
    # resnet152 = resnet_152()
    # resnet18.build(input_shape=(None,224,224,3))
    # resnet18.summary()

    x = tf.random.uniform([1,224,224,3])
    output1 = resnet50(x,training=True)
    print(output1,output1.shape)


    # return

if __name__ == '__main__':
    main()