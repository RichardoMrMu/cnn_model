# -*- coding: utf-8 -*-
# @Time    : 2020-12-02 19:15
# @Author  : RichardoMu
# @File    : UNET.py
# @Software: PyCharm
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow as tf


base_model = tf.keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)
# 使用这些层的feature
for i,layer in enumerate(base_model.layers):
    print(i,layer.name)
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]
print(layers)
down_stack = tf.keras.Model(inputs=base_model.input,outputs=layers)
down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512,3),
    pix2pix.upsample(256,3),
    pix2pix.upsample(128,3),
    pix2pix.upsample(64,3)
]


def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs

    # 在模型中降频取样
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # 升频取样然后建立跳跃连接
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # 这是模型的最后一层
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)