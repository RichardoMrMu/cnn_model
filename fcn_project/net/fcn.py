# -*- coding: utf-8 -*-
# @Time    : 2020-12-01 19:00
# @Author  : RichardoMu
# @File    : fcn.py
# @Software: PyCharm

import tensorflow as tf
import tensorflow.keras as k

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FCN(Model):

    def __init__(self, n_class, rate):
        super(FCN, self).__init__(name="FCN")

        self.relu = tf.keras.layers.ReLU()

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        self.convolization1 = tf.keras.layers.Conv2D(filters=4096, kernel_size=(1, 1), strides=(1, 1), padding='SAME')
        self.convolization2 = tf.keras.layers.Conv2D(filters=4096, kernel_size=(1, 1), strides=(1, 1), padding='SAME')
        self.convolization3 = tf.keras.layers.Conv2D(filters=n_class, kernel_size=(1, 1), strides=(1, 1), padding='SAME')

        self.up1 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='SAME')
        self.conv1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='SAME')

        self.up2 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='SAME')
        self.conv2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='SAME')

        self.up3 = tf.keras.layers.Conv2DTranspose(filters=n_class, kernel_size=(16, 16), strides=(8, 8), padding='SAME')

    def call(self, inputs):
        h = self.convolization1(inputs[-1])
        h = self.relu(self.dropout1(h))
        h = self.convolization2(h)
        h = self.relu(self.dropout2(h))
        h = self.convolization3(h)

        h = self.up1(h)
        x_32 = self.relu(self.conv1(inputs[-2]))
        h = tf.math.add(h, x_32)

        h = self.up2(h)
        x_16 = self.relu(self.conv2(inputs[-3]))
        h = tf.math.add(h, x_16)

        self.out = self.up3(h)

        return self.out
def get_base_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)
    for i,layer in enumerate(base_model.layers):
        print(i,layer.name)
    tf.keras.utils.plot_model(base_model,show_shapes=True)
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    print(base_model.get_layer(layer_names[1]).output)
    layers = [base_model.get_layer(name).output for name in layer_names]
    down_stack = tf.keras.Model(inputs=base_model.input,outputs=layers)
    down_stack.trainable = False
    return down_stack
# def fcn_model():
#     fcn = FCN()
#     inputs = tf.keras.layers.Input(shape=[128,128,3])
#     x = inputs
#
#     down_stack = get_base_model()
#     skips = down_stack(x)
#     x = skips[-1]
#     skips = tf.reverse(skips)
#     # 升频取样然后建立跳跃连接
#     for up, skip in zip(upt)
#     return



class FCN32s(tf.keras.Model):
    def __init__(self,pretrained_net,num_class):
        super(FCN32s, self).__init__()
        self.n_class = num_class
        self.pretrained_net = pretrained_net
        self.relu = tf.keras.layers.ReLU()
        self.deconv_bn_1 = tf.keras.Sequential([
            tf.keras.layers.Convolution2DTranspose(filters=512,kernel_size=3,strides=2,
                                                   padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.deconv_bn_2 = tf.keras.Sequential([
            tf.keras.layers.Convolution2DTranspose(filters=256,kernel_size=3,strides=2,
                                                   padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.deconv_bn_3 = tf.keras.Sequential([
            tf.keras.layers.Convolution2DTranspose(filters=128, kernel_size=3, strides=2,
                                                   padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.deconv_bn_4 = tf.keras.Sequential([
            tf.keras.layers.Convolution2DTranspose(filters=64, kernel_size=3, strides=2,
                                                   padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.deconv_bn_5 = tf.keras.Sequential([
            tf.keras.layers.Convolution2DTranspose(filters=32, kernel_size=3, strides=2,
                                                   padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.classifier = tf.keras.layers.Conv2D(filters=num_class,kernel_size=1,padding='same')
    # def get_deconv_layers(self):
    #     layers = []
    #     for i
    # def get_deconv_bn(self,filter,kernel_size=3,strides=2,padding='same'):
    #     deconv_bn = tf.keras.Sequential([
    #         tf.keras.layers.Convolution2DTranspose(
    #             filter=filter, kernel_size=3, strides=2, padding='same'
    #         ),
    #         tf.keras.layers.BatchNormalization()
    #     ])
    #     return deconv_bn

    def call(self,input,training=None):
        output = self.pretrained_net(input)
        score = self.deconv_bn_1(output,training=training)  # size=(N, x.H/16, x.W/16, 512)
        score = self.deconv_bn_2(score,training=training)  # size=(N, x.H/8, x.W/8, 256)
        score = self.deconv_bn_3(score,training=training)  # size=(N, x.H/4, x.W/4, 128)
        score = self.deconv_bn_4(score,training=training)  # size=(N, x.H/2, x.W/2, 64)
        score = self.deconv_bn_5(score,training=training)  # size=(N, x.H, x.W, 32)
        score = self.classifier(score,training=training)   # size=(N, x.H/1, x.W/1, n_class)
        return score # size=(N, x.H/1, x.W/1, n_class)

def get_fcn32():
    base_model = tf.keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False,weights='imagenet')
    for index , layer in enumerate(base_model.layers):
        print(index,layer.name)
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs
    fcn32 = FCN32s(base_model,3)
    y = fcn32(x)
    return tf.keras.Model(inputs=inputs, outputs=y)



def main():
    get_base_model()
if __name__ == '__main__':
    main()