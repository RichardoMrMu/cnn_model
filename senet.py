# -*- coding: utf-8 -*-
# @Time    : 2020-11-12 10:20
# @Author  : RichardoMu
# @File    : senet.py
# @Software: PyCharm

import tensorflow as tf
import os
from se_layer import _make_bottleneck_layer
os.environ["TF_CPP_MIN_LOG_LEVLE"] = '2'
os.environ['CUDA_VISIBEL_DEVICES'] = '3'
# set gpu growth step by step
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
# set random seed
tf.random.set_seed(1234)

class SEResNet(tf.keras.Model):
    def __init__(self,layers_param,num_class=10):
        super(SEResNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,kernel_size=7,strides=2,padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')
        self.layer1 = _make_bottleneck_layer(64,layers_param[0],2)
        self.layer2 = _make_bottleneck_layer(128,layers_param[1],2)
        self.layer3 = _make_bottleneck_layer(256,layers_param[2],2)
        self.layer4 = _make_bottleneck_layer(512,layers_param[3],2)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(1000,activation='relu')
        self.fc2 = tf.keras.layers.Dense(num_class)


    def call(self,input_x,training=None):
        output = self.conv1(input_x)
        output = self.bn1(output,training=training)
        output = self.max_pool(output)
        output = self.layer1(output)
        output = self.layer2(output,training=training)
        output = self.layer3(output,training=training)
        output = self.layer4(output,training=training)
        output = self.avg_pool(output)
        output = self.fc(output)
        output = self.fc2(output)
        return output


def se_resnet_50():
    return  SEResNet(layers_param=[3,4,6,3],num_class=10)


def se_resnet_101():
    return SEResNet(layers_param=[3,4,23,3],num_class=10)


def se_resnet_152():
    return SEResNet(layers_param=[3,8,36,3],num_class=10)


def main():
    model = se_resnet_50()
    input_x = tf.random.uniform([3,224,224,3])
    output = model(input_x)
    print(f"shape of output :{output.shape},output:{output}")
    return

if __name__ == '__main__':
    main()