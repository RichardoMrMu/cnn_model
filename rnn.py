# -*- coding: utf-8 -*-
# @Time    : 2020-11-10 21:59
# @Author  : RichardoMu
# @File    : rnn.py
# @Software: PyCharm
"""
rnn分析imdb的情感分析
"""
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
def preprocess(x,y):
    y = tf.one_hot(y,depth=2)
    print(y)
    return x,y
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# 设置评价的句子长度
max_review_length = 80
# 设置词汇列表，只使用最常用的10000个，剩下的归零
top_words = 10000
batchsz = 32
# load data
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.imdb.load_data(num_words=top_words)
print(len(x_train[0]),y_train[0])
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_review_length)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_review_length)
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_db = train_db.shuffle(1000).batch(batchsz,drop_remainder=True)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.batch(batchsz,drop_remainder=True)
x,y = next(iter(train_db))
print(x.shape,y.shape,y)

def plot_graph(history,metrics):
    plt.plot(history.history[metrics])
    plt.plot(history.history['val_' + metrics], '')
    plt.xlabel("Epochs")
    plt.ylabel(metrics)
    plt.legend([metrics, 'val_' + metrics])
    plt.show()

class RNN(tf.keras.Model):
    def __init__(self,units ,num_class,num_layers):
        super(RNN, self).__init__()
        # supter(RNN,self).__init__()
        self.rnn = tf.keras.layers.LSTM(units,return_sequences=True)
        self.rnn2 = tf.keras.layers.LSTM(units)

        self.embedding = tf.keras.layers.Embedding(top_words,100,input_length=max_review_length)
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1)
        ])

    def call(self,inputs , training=None,mask=None):

        x = self.embedding(inputs)
        x = self.rnn(x)
        x = self.rnn2(x)

        x = self.fc(x)

        print(x.shape)
        print(x)
        return x

def main():
    tf.random.set_seed(1234)
    units = 64
    num_classes = 2
    batch_size = 32
    epochs = 5
    model = RNN(units,num_class=num_classes,num_layers=2)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(x=x_train,y=y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test,y_test),verbose=1)

    scores = model.evaluate(x_test,y_test,batch_size,verbose=1)
    print(f"finale test loss and accuracy:{scores}")
    plot_graph(history,'accuracy')
    plot_graph(history,'loss')
if __name__ == '__main__':
    main()