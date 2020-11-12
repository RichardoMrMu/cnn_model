# -*- coding: utf-8 -*-
# @Time    : 2020-11-10 10:29
# @Author  : RichardoMu
# @File    : resnet_block.py
# @Software: PyCharm

import tensorflow as tf


class BaseBlocks(tf.keras.Model):
    def __init__(self,filter_num,strides=1):
        super(BaseBlocks, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,kernel_size=(3,3),strides=strides,padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,kernel_size=(3,3),strides=1,padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        if strides != 1:
            self.downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=filter_num,kernel_size=(1,1),strides=strides),
                tf.keras.layers.BatchNormalization()
            ])
        else:
            self.downsample = lambda x:x

    def call(self,x,training=None):
        residual = self.downsample(x)

        output = self.conv1(x)
        output = self.bn1(output,training=training)
        output = tf.nn.relu(output)
        output = self.conv2(output)
        output = self.bn2(output,training=training)
        output = tf.nn.relu(tf.keras.layers.add([residual,output]))
        return output


class BottleNeckBlocks(tf.keras.Model):
    def __init__(self,filter_num,strides=1):
        super(BottleNeckBlocks, self).__init__()
        self.layer1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filter_num,kernel_size=1,strides=1,padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.layer2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filter_num,kernel_size=3,strides=strides,padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.layer3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filter_num * 4, kernel_size=1, strides=1, padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.downsample = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filter_num*4,kernel_size=1,strides=strides,padding='same'),
            tf.keras.layers.BatchNormalization()
        ])

    def call(self,x,training=None):
        residual = self.downsample(x)

        output = self.layer1(x,training=training)
        output = tf.nn.relu(output)
        output = self.layer2(output,training=training)
        output = tf.nn.relu(output)
        output = self.layer3(output,training=training)
        output = tf.nn.relu(tf.keras.layers.add([output,residual]))
        return output


def _make_basic_block_layer(filter_num,blocks,strides=1):
    res_block = tf.keras.Sequential()
    res_block.add(BaseBlocks(filter_num,strides=strides))
    for _ in range(1,blocks):
        res_block.add(BaseBlocks(filter_num,strides=1))
    return res_block


def _make_bottleneck_layer(filter_num,blocks,strides=1):
    res_bottleneck_block = tf.keras.Sequential()
    res_bottleneck_block.add(BottleNeckBlocks(filter_num,strides=strides))
    for _ in range(1,blocks):
        res_bottleneck_block.add(BottleNeckBlocks(filter_num,strides=1))
    return res_bottleneck_block

