# -*- coding: utf-8 -*-
# @Time    : 2020-12-01 19:42
# @Author  : RichardoMu
# @File    : batch.py
# @Software: PyCharm

import os, cv2
import numpy as np
import random as rd

from tensorflow.keras.applications.vgg19 import preprocess_input


def load_path_list(image_path, gt_path, batch_size, train=True):
    """
    Load all path of image, there is two case in this code
    if train == True then load training image
    training image calculate there all size and calculate it's validation size
    if train == False then load test image
    doesn't calculate anythings
    path_list : Data File Name List
    Train_size : int
    Validation_size : int
    """

    if train:
        print("Image Load Started..")

        path_list = os.listdir(gt_path)

        image_size = len(path_list)
        Train_size = image_size // batch_size * batch_size
        Validation_size = image_size - Train_size

        if Validation_size < 10:
            Train_size -= batch_size
            Validation_size += batch_size

        print("Train data size : ", Train_size)
        print("Validation data size : ", Validation_size)
    else:
        path_list = os.listdir(gt_path)
        Train_size = 0
        Validation_size = 0
        print("Test data size : ", len(path_list))

    rd.shuffle(path_list)

    return path_list, Train_size, Validation_size


def load_labels(label_path):
    """
    Load labels for VOC2012, Label must be maded txt files and like my label.txt
    Label path can be change when run training code , use --label_path
    label : { label naem : label color}
    index : [ [label color], [label color]]
    """

    with open(label_path, "r") as f:
        lines = f.readlines()

    label = {}
    index = []
    for line in lines:
        sp = line.split()
        label[sp[0]] = [int(sp[1]), int(sp[2]), int(sp[3])]
        index.append([int(sp[3]), int(sp[2]), int(sp[1])])

    return label, index


def make_label_map(path, label_list):
    """
    make 3D ground Truth image to 1D Labeled image
    Images has multi label on each point and I removed last label
    Output : [N, H, W]
    """

    img = []
    for name in path:
        now = np.zeros((256, 256))
        im = cv2.resize(cv2.imread(name), (256, 256)).tolist()
        for y, i in enumerate(im):
            for x, j in enumerate(i):
                try:
                    now[y, x] = label_list.index(j)

                except ValueError:
                    now[y, x] = 0

        img.append(now)
    return img


def image_load_resize(path):
    """
    make image to 256 * 256 3D image
    Output : [N, H, W, C]
    """

    img = []
    for name in path:
        img.append(cv2.resize(cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB), (256, 256)))

    return preprocess_input(np.array(img))


def batch(img_path, gt_path, img_list, batch_size, label_list):
    """
    Batch Main function
    Return image and Label Map
    Output : [N, H, W, C], [N, H, W]
    """

    image_list = [os.path.join(img_path, i) for i in img_list]
    gt_list = [os.path.join(gt_path, i) for i in img_list]

    for i in range(0, len(img_list), batch_size):
        yield image_load_resize(image_list[i:i + batch_size]), make_label_map(gt_list[i:i + batch_size], label_list)