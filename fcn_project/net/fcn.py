# -*- coding: utf-8 -*-
# @Time    : 2020-12-01 19:00
# @Author  : RichardoMu
# @File    : fcn.py
# @Software: PyCharm

import tensorflow as tf
import tensorflow.keras as k

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


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