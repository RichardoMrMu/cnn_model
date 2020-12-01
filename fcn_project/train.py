# -*- coding: utf-8 -*-
# @Time    : 2020-12-01 19:22
# @Author  : RichardoMu
# @File    : train.py
# @Software: PyCharm


import os, sys, cv2, glob, argparse

import numpy as np
import random as rd

import tensorflow as tf
import tensorflow.keras as k

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from net import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19 as vgg19
from tensorflow.keras.applications.vgg19 import preprocess_input

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(
    usage="""train.py --train_dir TrainDirectory --train_gt_dir TrainGroundTruthDirectory --test.py TestDirectory --test_gt_dir TestGroundTruthDirectory [Option]""")
parser.add_argument("--train_dir", help="Train directory Location")
parser.add_argument("--train_gt_dir", help="Train ground truth Location")
parser.add_argument("--batch_size", nargs="?", default=8, help="Number of batch_size")
parser.add_argument("--label", default="../Data/VOC2012/labels.txt", help="location of labels")
parser.add_argument("--epoch", nargs="?", default=20, help="Number of Training epochs")
parser.add_argument("--lr", nargs="?", default=1e-4, help="Number of learning rate")
parser.add_argument("--lr_decay", nargs="?", default=5e-5, help="Number of learning rate decay")
parser.add_argument("--rate", nargs="?", default=0.5, help="Number of drop out rate")
parser.add_argument("--save_dir", nargs="?", default="./model/", help="Location of saved model directory")

args = parser.parse_args()

train_path = args.train_dir
train_gt_path = args.train_gt_dir
save_path = args.save_dir
label_path = args.label
rate = float(args.rate)
lr = float(args.lr)
lr_decay = float(args.lr_decay)
epochs = int(args.epoch)
batch_size = int(args.batch_size)

if not os.path.exists(save_path): os.mkdir(save_path)
if not os.path.exists(os.path.join(".", "result")): os.mkdir(os.path.join(".", "result"))
if not os.path.exists(os.path.join(".", "result_img")): os.mkdir(os.path.join(".", "result_img"))
if not os.path.exists(os.path.join(".", "result_img", "val")): os.mkdir(os.path.join(".", "result_img", "val"))
if not os.path.exists(train_path): raise TypeError("Please input right Train Data Path")
if not os.path.exists(train_gt_path): raise TypeError("Please input right Train Ground Truth Data Path")


def train_step(features, gt):
    with tf.GradientTape() as tape:
        pred = model(features, training=True)
        loss = tf.keras.losses.categorical_crossentropy(gt, pred, from_logits=True)

    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    print("에포크 : ", epoch, " 스텝 : ", step, ", Loss : ", float(tf.reduce_mean(loss)))


trlist, t_size, val_size = load_path_list(train_path, train_gt_path, batch_size)
trlist, valist = (trlist[:t_size], trlist[t_size: t_size + val_size])

labels, index = load_labels(label_path)
val_bathces = batch(train_path, train_gt_path, valist, batch_size, index)
val_img, val_gt = next(val_bathces)
n_class = len(labels)

FCN_Layer = ['block3_pool', 'block4_pool', 'block5_pool']

encoder = pre_vgg(FCN_Layer)
model = FCN(n_class, rate)
lr_schedule = k.optimizers.schedules.ExponentialDecay(lr, decay_steps=1, decay_rate=lr_decay, staircase=False)
optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

for epoch in range(epochs):

    rd.shuffle(trlist)
    batches = batch(train_path, train_gt_path, trlist, batch_size, index)

    for step, now in enumerate(batches):

        train_img, gt_img = now

        gt_img = k.utils.to_categorical(gt_img, n_class)
        features = encoder(train_img)

        train_step(features, gt_img)

        if step % 100 == 0:
            val_features = encoder(val_img)
            val_pred = model(val_features, training=False)
            val_pred = tf.math.argmax(val_pred, axis=3)

            for i, image in enumerate(val_pred):
                file_name = str(epoch) + " epoch_" + str(step) + " step_" + str(i) + ".jpg"
                plt.imsave(os.path.join(".", "result_img", "val", file_name), image, cmap=cm.Paired)