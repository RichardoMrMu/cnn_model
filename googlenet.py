# -*- coding: utf-8 -*-
# @Time    : 2020-11-03 20:09
# @Author  : RichardoMu
# @File    : googlenet.py
# @Software: PyCharm
"""
inception version 1
"""
import tensorflow as tf
import os
from tensorflow.keras.regularizers import l2 as L2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


class ConvBNRelu(tf.keras.Model):
    def __init__(self,filter,kernelsz=3,strides=1,padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filter,kernelsz,strides=strides,padding=padding, kernel_regularizer=L2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self,x,training=None):
        x = self.model(x,training=training)
        return x


class InceptionBlk(tf.keras.Model):
    def __init__(self,filters_1x1,filters_3x3_reduce,filters_3x3,filters_5x5,filters_5x5_reduce,filters_pool,strides=1):
        super(InceptionBlk, self).__init__()
        self.conv_1x1 = tf.keras.Sequential([
          ConvBNRelu(filters_1x1,kernelsz=1,strides=strides)
        ])

        self.conv_3x3_withreduce = tf.keras.Sequential([
            ConvBNRelu(filters_3x3_reduce,kernelsz=1,strides=strides),
            ConvBNRelu(filters_3x3,kernelsz=3,strides=strides)
        ])
        self.conv_5x5_withreduce = tf.keras.Sequential([
            ConvBNRelu(filters_5x5,kernelsz=1,strides=strides),
            ConvBNRelu(filters_5x5_reduce,kernelsz=5,strides=strides)
        ])
        self.maxpool_proj = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same'),
            ConvBNRelu(filters_pool,kernelsz=1,strides=strides)
        ])

    def call(self,x,training=None):
        x1_1 = self.conv_1x1(x,training=training)
        x3_3 = self.conv_3x3_withreduce(x,training=training)
        x5_5 = self.conv_5x5_withreduce(x,training=training)
        x_maxpoll = self.maxpool_proj(x,training=training)
        # concat along axis = channel
        x = tf.concat([x1_1,x3_3,x5_5,x_maxpoll],axis=3)
        return x


class GoogleNet(tf.keras.Model):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.layer1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64,kernel_size=7,strides=2,padding='same',activation='relu',kernel_regularizer=L2(0.01)),
            tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=64,kernel_size=1,strides=1,padding='same',activation='relu',kernel_regularizer=L2(0.01)),
            tf.keras.layers.Conv2D(filters=192,kernel_size=3,strides=1,padding='same',activation='relu',kernel_regularizer=L2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='same')
        ])
        self.layer2 = tf.keras.Sequential([
            InceptionBlk(filters_1x1=64,filters_3x3_reduce=96,filters_3x3=128,filters_5x5=32,filters_5x5_reduce=16,filters_pool=32),
            InceptionBlk(filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5=96, filters_5x5_reduce=32,
                         filters_pool=64),
        ])
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='same')
        self.layer3 = tf.keras.Sequential([
            InceptionBlk(filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208, filters_5x5=48,
                         filters_5x5_reduce=16,
                         filters_pool=64),
            InceptionBlk(filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224, filters_5x5=64,
                         filters_5x5_reduce=24,
                         filters_pool=64),
            InceptionBlk(filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256, filters_5x5=64,
                         filters_5x5_reduce=24,
                         filters_pool=64),
            InceptionBlk(filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288, filters_5x5=64,
                         filters_5x5_reduce=32,
                         filters_pool=64),
            InceptionBlk(filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5=128,
                         filters_5x5_reduce=32,
                         filters_pool=128),
        ])
        self.max_pool2 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.layer4 = tf.keras.Sequential([
            InceptionBlk(filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5=128,
                         filters_5x5_reduce=32,
                         filters_pool=128),
            InceptionBlk(filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384, filters_5x5=128,
                         filters_5x5_reduce=48,
                         filters_pool=128),
        ])
        self.layer5 = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D(pool_size=7, strides=1, padding='same'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1000),
            tf.keras.layers.Dense(10)
        ])

    def call(self,x,training=None):
        out = self.layer1(x,training=training)
        out = self.layer2(out,training=training)
        out = self.max_pool(out)
        out = self.layer3(out,training=training)
        out = self.max_pool2(out)
        out = self.layer4(out,training=training)
        out = self.layer5(out,training=training)
        return out


def main():
    model = GoogleNet()
    x = tf.random.uniform([10,224,224,3])
    output = model(x,training=True)
    print(output.shape,output)
    return


if __name__ == '__main__':
    main()
