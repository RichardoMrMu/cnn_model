# -*- coding: utf-8 -*-
# @Time    : 2020-12-01 19:42
# @Author  : RichardoMu
# @File    : utils.py
# @Software: PyCharm

import tensorflow as tf
import tensorflow.keras as k

from tensorflow.keras.applications import VGG19 as vgg19
from tensorflow.keras.applications.vgg19 import preprocess_input


def pre_vgg(layers):
    vgg = vgg19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layers]
    model = k.Model([vgg.input], outputs)

    return model
