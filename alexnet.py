# -*- coding: utf-8 -*-
# @Time    : 2020-10-30 15:26
# @Author  : RichardoMu
# @File    : alexnet.py
# @Software: PyCharm
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
class AlexNet(tf.keras.Model):
    def __init__(self,num_class=10):
        super(AlexNet, self).__init__()
        self.num_class = num_class
        self.create_model()

    def create_model(self):
        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11),strides=4, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),padding='same')
        ])
        self.conv2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', strides=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))
        ])
        self.conv3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', strides=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.conv4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', strides=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.conv5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)
        ])
        self.fc6 = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(rate=0.6)
        ])
        self.fc7 = tf.keras.Sequential([
            tf.keras.layers.Dense(4096),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(rate=0.6)
        ])
        self.fc8 = tf.keras.Sequential([

            tf.keras.layers.Dense(1000),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(rate=0.6)
        ])
        self.fc9 = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_class)
        ])


    def call(self,input_x):
        x = self.conv1(input_x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        return x

if __name__ == '__main__':
    input_tensor = tf.random.uniform([10,227,227,3])
    model = AlexNet(num_class=10)
    input_variable = tf.Variable(input_tensor)
    output = model(input_variable)
    print(output.shape)