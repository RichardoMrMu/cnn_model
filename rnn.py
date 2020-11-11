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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# 设置评价的句子长度
max_review_length = 80
# 设置词汇列表，只使用最常用的10000个，剩下的归零
top_words = 10000
# load data
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.imdb.load_data(num_words=top_words)
print(len(x_train[0]),y_train[0])
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_review_length)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_review_length)

class RNN(tf.keras.Model):
    def __init__(self,units ,num_class,num_layers):
        super(RNN, self).__init__()
        # supter(RNN,self).__init__()
        self.rnn = tf.keras.layers.LSTM(units,return_sequences=True)
        self.rnn2 = tf.keras.layers.LSTM(units)

        self.embedding = tf.keras.layers.Embedding(top_words,100,input_length=max_review_length)
        self.fc = tf.keras.layers.Dense(1)

    def call(self,inputs , training=None,mask=None):

        x = self.embedding(inputs)
        x = self.rnn(x)
        x = self.rnn2(x)

        x = self.fc(x)

        print(x.shape)
        return x

def main():
    tf.random.set_seed(1234)
    units = 64
    num_classes = 2
    batch_size = 32
    epochs = 2
    model = RNN(units,num_class=num_classes,num_layers=2)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(x=x_train,y=y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test,y_test),verbose=1)

    scores = model.evaluate(x_test,y_test,batch_size,verbose=1)
    print(f"finale test loss and accuracy:{scores}")
if __name__ == '__main__':
    main()