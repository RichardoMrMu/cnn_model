# -*- coding: utf-8 -*-
# @Time    : 2020-10-23 15:52
# @Author  : RichardoMu
# @File    : hand_classification.py
# @Software: PyCharm

import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
lr = 1e-3


(x,y) ,(x_val ,y_cal ) = tf.keras.datasets.mnist.load_data()
x = 2*tf.convert_to_tensor(x,dtype=tf.float32)/255.-1
y = tf.convert_to_tensor(y,dtype=tf.int32)
print(x.shape,y.shape)
y = tf.one_hot(y,depth=10)
print(x.shape,y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x,y)).batch(64)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10),
])
optimizer = tf.keras.optimizers.Adam(lr=lr)

def train(epoch):
    for step , (x,y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            x = tf.reshape(x,(-1,28*28))
            out = model(x)
            # y_onehot = tf.one_hot(y,depth=10)
            loss = tf.square(out-y)
            loss = tf.reduce_sum(loss)/x.shape[0]
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        if step % 100 == 0:
            print(f"epoch:{epoch},step:{step},loss:{loss.numpy()}")

def main():
    EPOCH = 10
    for i in range(EPOCH):
        train(i)

if __name__ == '__main__':
    main()
