# -*- coding: utf-8 -*-
# @Time    : 2020-11-12 10:54
# @Author  : RichardoMu
# @File    : se_layer.py
# @Software: PyCharm

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
tf.random.set_seed(1234)

class SELayer(tf.keras.layers.Layer):
    def __init__(self,input_channel,r=16):
        super(SELayer, self).__init__()
        # avg-pool
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(units=input_channel//r,activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=input_channel,activation='sigmoid')

    def call(self,input_x):
        branch = self.avg_pool(input_x)
        branch = self.fc1(branch)
        branch = self.fc2(branch)
        # branck -> [b,input_channel] -> [b,1,1,input_channel]
        branch = tf.expand_dims(branch,axis=1)
        branch = tf.expand_dims(branch,axis=1)
        return tf.keras.layers.multiply([input_x,branch])


class BottleNeckBlocks(tf.keras.Model):
    def __init__(self,filter_num,strides=1):
        super(BottleNeckBlocks,self).__init__()
        self.layer1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filter_num,
                                   kernel_size=1,
                                   strides=1,
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.layer2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filter_num,
                                   kernel_size=3,
                                   strides=strides,
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.layer3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filter_num * 4,
                                   kernel_size=1,
                                   strides=1,
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.downsample = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filter_num * 4,
                                   kernel_size=1,
                                   strides=strides,
                                   padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.se = SELayer(input_channel=filter_num * 4)

    def call(self,input_x,training=None):
        residual = self.downsample(input_x)

        out = self.layer1(input_x,training=training)
        out = self.layer2(out,training=training)
        out = self.layer3(out,training=training)
        out = self.se(out)

        out = tf.keras.layers.add([out,residual])

        return tf.nn.relu(out)


def _make_bottleneck_layer(filter_num,blocks,strides=1):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeckBlocks(filter_num, strides=strides))
    for _ in range(1, blocks):
        res_block.add(BottleNeckBlocks(filter_num, strides=1))
    return res_block

#
# class NormalDense(tf.keras.Model):
#     def __init__(self):
#         super(NormalDense, self).__init__()
#         self.dense1 = tf.keras.layers.Dense(1000)
#         self.dense2 = tf.keras.layers.Dense(32)
#         self.dense3 = tf.keras.layers.Dense(1)
#
#     def call(self,inputs):
#         out = self.dense1(inputs)
#         out = self.dense2(out)
#         out = self.dense3(out)
#         return out
#
#
# class DenseWithFlatten(tf.keras.Model):
#     def __init__(self):
#         super(DenseWithFlatten, self).__init__()
#         self.flatten1 = tf.keras.layers.Flatten()
#         self.dense1 = tf.keras.layers.Dense(1000)
#         self.dense2 = tf.keras.layers.Dense(32)
#         self.dense3 = tf.keras.layers.Dense(1)
#
#     def call(self,inputs):
#         out = self.flatten1(inputs)
#         out = self.dense1(out)
#         out = self.dense2(out)
#         out = self.dense3(out)
#         return out
#

def main():

    input_x = tf.random.uniform([3,4,4,3])
    # model1 = DenseWithFlatten()
    # model2 = NormalDense()
    # output1 = model1(input_x)
    # output2 = model2(input_x)
    # print(output1.shape,output2.shape)
    # print(f"output with flatten :{output1},output without flatten:{output2}")

    return


if __name__ == '__main__':

    main()


