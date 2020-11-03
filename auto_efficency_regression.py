# -*- coding: utf-8 -*-
# @Time    : 2020-10-27 9:34
# @Author  : RichardoMu
# @File    : auto_efficency_regression.py
# @Software: PyCharm


import tensorflow as tf
import pandas as pd
import os
from matplotlib import pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# get data
dataset_path = tf.keras.utils.get_file("auto-mpg.data",
                                       "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)
# params
lr = 1e-3
# 利用pandas读取数据
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
'Acceleration', 'Model Year', 'Origin']
raw_data = pd.read_csv(dataset_path,names=column_names,
                       na_values="?",comment="\t",
                       sep=" ",skipinitialspace=True)
dataset = raw_data.copy()
print(dataset.head())
# 统计空白数据
print(dataset.isna().sum())
origin = dataset.pop('Origin')
# 根据origin列来写入3个新的列
dataset['USA'] = (origin==1) * 1.0
dataset['Europe'] = (origin==2) * 1.0
dataset['Japan'] = (origin==3) * 1.0
print(dataset.tail())

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
print(train_stats.columns.values)
train_stats.pop("MPG")
train_stats = train_stats.transpose()

train_label = train_dataset.pop("MPG")
test_label = test_dataset.pop("MPG")


def norm(x):
    return (x-train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = tf.keras.layers.Dense(64,activation='relu')
        self.fc2 = tf.keras.layers.Dense(64,activation='relu')
        self.fc3 = tf.keras.layers.Dense(1)
    def call(self,inputs,training=None,mask=None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = Network()
model.build(input_shape=(None,9))
model.summary()
optimizer = tf.keras.optimizers.RMSprop(lr=lr)
train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values,train_label.values))
train_db = train_db.shuffle(100).batch(32)

train_losses = []
test_losses = []

def train(epoch):
    for step ,(x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = model(x)
            loss = tf.reduce_mean(tf.keras.losses.MSE(y,out))
            mae_loss = tf.reduce_mean(tf.losses.MAE(y,out))
        if step % 10 == 0:
            print(f"epoch:{epoch},step:{step},loss:{loss}")
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
    train_losses.append(float(mae_loss))
    out = model(tf.constant(normed_test_data.values))
    test_losses.append(tf.reduce_mean(tf.losses.MAE(test_label,out)))

def main():
    epochs = 200
    for i in range(epochs):
        train(epoch=i)
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("MAE")
    plt.plot(train_losses,label='Train')

    plt.plot(test_losses, label='Test')
    plt.legend()
    plt.savefig('auto.svg')
    plt.show()

if __name__ == '__main__':
    main()