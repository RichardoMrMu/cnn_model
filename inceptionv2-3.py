# -*- coding: utf-8 -*-
# @Time    : 2020-11-09 19:34
# @Author  : RichardoMu
# @File    : inceptionv2-3.py
# @Software: PyCharm
from googlenet import ConvBNRelu
import tensorflow as tf


class InceptionBlkA(tf.keras.Model):
    def __init__(self,filters):
        super(InceptionBlkA, self).__init__()
        self.conv_1x1_3x3_3x3 = tf.keras.Sequential([
            ConvBNRelu(filter=filters,kernelsz=1,strides=1),
            ConvBNRelu(filter=filters,kernelsz=3,strides=1),
            ConvBNRelu(filter=filters,kernelsz=3,strides=1)
        ])
        self.conv_1x1_3x3 = tf.keras.Sequential([
            ConvBNRelu(filter=filters, kernelsz=1, strides=1),
            ConvBNRelu(filter=filters, kernelsz=3, strides=1),
        ])
        self.pool_1x1 = tf.keras.Sequential([
            tf.keras.layers.AvgPool2D(pool_size=3,strides=1,padding='same'),
            ConvBNRelu(filter=filters,kernelsz=1,strides=1)
        ])
        self.conv_1x1 = ConvBNRelu(filter=filters,kernelsz=1,strides=1)

    def call(self,x,training=None):
        output1 = self.conv_1x1_3x3_3x3(x,training=training)
        output2 = self.conv_1x1_3x3(x,training=training)
        output3 = self.conv_1x1_3x3_3x3(x,training=training)
        output4 = self.pool_1x1(x,training=training)
        output = tf.concat([output1,output2,output3,output4],axis=3)
        return output


class InceptionBlkB(tf.keras.Model):
    def __init__(self, filters):
        super(InceptionBlkB, self).__init__()
        self.conv_1x1_3x3_3x3 = tf.keras.Sequential([
            ConvBNRelu(filter=filters, kernelsz=1, strides=1),
            ConvBNRelu(filter=filters, kernelsz=3, strides=1),
            ConvBNRelu(filter=filters, kernelsz=3, strides=1)
        ])
        self.conv_1x1_3x3 = tf.keras.Sequential([
            ConvBNRelu(filter=filters, kernelsz=1, strides=1),
            ConvBNRelu(filter=filters, kernelsz=3, strides=1),
        ])
        self.pool_1x1 = tf.keras.Sequential([
            tf.keras.layers.AvgPool2D(pool_size=3, strides=1, padding='same'),
            ConvBNRelu(filter=filters, kernelsz=1, strides=1)
        ])
        self.conv_1x1 = ConvBNRelu(filter=filters, kernelsz=1, strides=1)

    def call(self, x, training=None):
        output1 = self.conv_1x1_3x3_3x3(x, training=training)
        output2 = self.conv_1x1_3x3(x, training=training)
        output3 = self.conv_1x1_3x3_3x3(x, training=training)
        output4 = self.pool_1x1(x, training=training)
        output = tf.concat([output1, output2, output3, output4], axis=3)
        return output