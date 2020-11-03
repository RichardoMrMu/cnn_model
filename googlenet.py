# -*- coding: utf-8 -*-
# @Time    : 2020-11-03 20:09
# @Author  : RichardoMu
# @File    : googlenet.py
# @Software: PyCharm
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


class Inception(tf.keras.Model):
    def __init__(self):
        super(Inception, self).__init__()
        

class GoogleNet(tf.keras.Model):
    def __init__(self):
        super(GoogleNet, self).__init__()
