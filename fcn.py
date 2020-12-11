# -*- coding: utf-8 -*-
# @Time    : 2020-11-12 16:08
# @Author  : RichardoMu
# @File    : fcn.py
# @Software: PyCharm
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class FCN32s(tf.keras.Model):
    def __init__(self,pretrained_net,num_class):
        super(FCN32s, self).__init__()
        self.n_class = num_class
        self.pretrained_net = pretrained_net
        self.relu = tf.keras.layers.ReLU()
        self.deconv_bn_1 = tf.keras.Sequential([
            tf.keras.layers.Convolution2DTranspose(filter=512,kernel_size=3,strides=2,
                                                   padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.deconv_bn_2 = tf.keras.Sequential([
            tf.keras.layers.Convolution2DTranspose(filter=256,kernel_size=3,strides=2,
                                                   padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.deconv_bn_3 = tf.keras.Sequential([
            tf.keras.layers.Convolution2DTranspose(filter=128, kernel_size=3, strides=2,
                                                   padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.deconv_bn_4 = tf.keras.Sequential([
            tf.keras.layers.Convolution2DTranspose(filter=64, kernel_size=3, strides=2,
                                                   padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.deconv_bn_5 = tf.keras.Sequential([
            tf.keras.layers.Convolution2DTranspose(filter=32, kernel_size=3, strides=2,
                                                   padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.classifier = tf.keras.layers.Conv2D(filter=num_class,kernel_size=1,padding='same')
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
        x5 = output['x5']
        score = self.deconv_bn_1(x5,training=training)  # size=(N, x.H/16, x.W/16, 512)
        score = self.deconv_bn_2(score,training=training)  # size=(N, x.H/8, x.W/8, 256)
        score = self.deconv_bn_3(score,training=training)  # size=(N, x.H/4, x.W/4, 128)
        score = self.deconv_bn_4(score,training=training)  # size=(N, x.H/2, x.W/2, 64)
        score = self.deconv_bn_5(score,training=training)  # size=(N, x.H, x.W, 32)
        score = self.classifier(score,training=training)   # size=(N, x.H/1, x.W/1, n_class)
        return score # size=(N, x.H/1, x.W/1, n_class)

def get_fcn32():
    base_model = tf.keras.applications.MobileNetV2(include_top=False,weights='imagenet')
    for index , layer in enumerate(base_model.layers):
        print(index,layer.name)
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs
    fcn32 = FCN32s(base_model,3)
    y = fcn32(x)
    return tf.keras.Model(inputs=inputs, outputs=y)

def main():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False,weights='imagenet')
    for index , layer in enumerate(vgg.layers):
        print(index,layer.name)
    inputs = tf.random.uniform([1,224,224,3])
    fcn = FCN32s()
    output = vgg(inputs,training=True)
    output = fcn(inputs,training=True)


if __name__ == '__main__':
    main()