# -*- coding: utf-8 -*-
# @Time    : 2020-11-02 16:39
# @Author  : RichardoMu
# @File    : vgg.py
# @Software: PyCharm

import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


class VGG(tf.keras.Model):
    def __init__(self):
        super(VGG, self).__init__()
        self.layer1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation=tf.keras.layers.LeakyReLU()),
            tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),

        ])
        self.layer2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128,(3,3,),padding='same',activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),
        ])
        self.layer3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256,(3,3),padding='same',activation='relu'),
            tf.keras.layers.Conv2D(256, (3, 3),padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),
        ])
        self.layer4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(512,(3,3),padding='same',activation='relu'),
            tf.keras.layers.Conv2D(512,(3,3),padding='same',activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),
        ])
        self.layer4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu'),
            tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        ])
        self.dense1 = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096,activation='relu'),
            tf.keras.layers.Dense(4096,activation='relu'),
            tf.keras.layers.Dense(1000,activation='relu'),
            tf.keras.layers.Dense(10,activation=None)
        ])
        # self.ConLayer = tf.keras.Sequential([
        #     self.layer1(),
        #     self.layer2(),
        #     self.layer3(),
        #     self.layer4()
        # ])
        # self.DenseLayer = tf.keras.Sequential([
        #     self.dense1()
        # ])


    def call(self,input_x):
        # print(input_x.shape)
        x = self.layer1(input_x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = self.dense1(x)
        # print(x.shape)
        return x

class VGG13(tf.keras.Model):
    def __init__(self):
        super(VGG13, self).__init__()
configs = {
    'A': [
        '3-64','M',
        '3-128','M',
        '3-256','3-256','M',
        '3-512','3-512','M',
        '3-512','3-512','M'
    ],
    'B': [
        '3-64','3-64','M',
        '3-128','3-128','M',
        '3-256','3-256','M',
        '3-512','3-512','M',
        '3-512','3-512','M'
    ],
    'C': [
        '3-64','3-64','M',
        '3-128','3-128','M',
        '3-256','3-256','1-256','M',
        '3-512', '3-512','1-512', 'M',
        '3-512', '3-512', '1-512', 'M',
    ],
    'D': [
        '3-64','3-64','M',
        '3-128','3-128','M',
        '3-256','3-256','3-256','M',
        '3-512', '3-512','3-512', 'M',
        '3-512', '3-512', '3-512', 'M',
    ] ,
    'E': [
        '3-64','3-64','M',
        '3-128','3-128','M',
        '3-256','3-256','3-256','3-256','M',
        '3-512', '3-512','3-512','3-512', 'M',
        '3-512', '3-512', '3-512', '3-512','M',
    ] ,

}

class VGGNet(tf.keras.Model):
    """ VGGNet Architecture"""
    def __init__(self,config,class_count,rate=0.4):
        super(VGGNet, self).__init__()
        self.config = config
        self.class_count = class_count
        self.rate = rate
        self.conv_net = self.get_conv_net()
        self.fc_net = self.get_fc_net()



    def get_conv_net(self):
        """
        return the convolutional layers of the network
        :return:
        """
        layers = []
        print(configs[self.config])
        for layer in configs[self.config]:
            if layer == "M":
                layers.append(tf.keras.layers.MaxPool2D((2,2),strides=2,padding='same'))
            else:
                layer = layer.split('-')
                kernel_size = int(layer[0])
                filters = int(layer[1])
                layers.append(tf.keras.layers.Conv2D(filters=filters,
                                                     kernel_size=kernel_size,
                                                     padding='same',kernel_initializer='he_normal'))
                layers.append(tf.keras.layers.BatchNormalization())
                layers.append(tf.keras.layers.LeakyReLU())
        return tf.keras.Sequential(layers)


    def get_fc_net(self):
        """
        :return:the fully connected layers of the network
        """
        layers = []
        layers.append(tf.keras.layers.Flatten())
        layers.append(tf.keras.layers.LeakyReLU())
        layers.append(tf.keras.layers.Dropout(rate=self.rate))
        layers.append(tf.keras.layers.Dense(4096,kernel_initializer='he_normal'))
        layers.append(tf.keras.layers.Dense(4096,kernel_initializer='he_normal'))
        layers.append(tf.keras.layers.Dense(1000,kernel_initializer='he_normal'))
        layers.append(tf.keras.layers.Dense(self.class_count,kernel_initializer='he_normal'))
        return tf.keras.Sequential(layers)


    def call(self,input_x,training=None):
        x = self.conv_net(input_x,training=training)
        x = self.fc_net(x,training=training)
        return x


def main():
    tf.random.set_seed(1234)
    input_x = tf.random.uniform([1,227,227,3])
    model = VGGNet('A',10)
    output = model(input_x,training=True)
    output2 = model(input_x)
    output1 = model(input_x,training=False)
    print(output)
    print(output2)
    print(output1)
    # print(output.shape)
    # for layer in model.layers:
    #     print(layer.summary())

if __name__ == '__main__':
    main()