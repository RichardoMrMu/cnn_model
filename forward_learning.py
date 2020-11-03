# -*- coding: utf-8 -*-
# @Time    : 2020-10-26 15:51
# @Author  : RichardoMu
# @File    : forward_learning.py
# @Software: PyCharm
import  tensorflow as tf
import os
from matplotlib import pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus :
    tf.config.experimental.set_memory_growth(gpu,True)
# params
lr = 1e-3
batchsz = 64
# get data


w1 = tf.Variable(tf.random.truncated_normal([784,256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256,128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128,10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

optimizer = tf.keras.optimizers.Adam(lr=lr)
losses = []
accs = []
def preprocess(x,y):
    print(x.shape,y.shape)
    x =tf.cast(x,dtype=tf.float32)
    x = tf.reshape(x,[-1,28*28])
    y = tf.cast(y,dtype=tf.int32)
    y = tf.one_hot(y,depth=10)
    return x,y

def getDataLoader():
    (x,y),(x_val,y_val) = tf.keras.datasets.mnist.load_data()
    print(f"x.shape:{x.shape},y.shape:{y.shape},x_val.shape:{x_val.shape},y_val.shape:{y_val.shape}")
    # transfer to tensor
    train_db = tf.data.Dataset.from_tensor_slices((x,y))
    # 缓冲区1000张shuffle
    train_db = train_db.shuffle(1000)
    train_db = train_db.batch(batchsz)
    train_db = train_db.map(preprocess)
    train_db = train_db.repeat(20)

    # get test database preprocess
    test_db = tf.data.Dataset.from_tensor_slices((x_val,y_val))
    test_db.shuffle(1000).batch(batchsz).map(preprocess)
    x,y = next(iter(train_db))
    print(x.shape,y.shape)
    return train_db,test_db

def train(epoch,train_db):
    for step , (x,y) in enumerate(train_db):

        with tf.GradientTape() as tape:
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2 @ w3 + b3
            loss = tf.square(y-out)
            loss = tf.reduce_mean(loss)
        # compute gradients
        grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
        # backward
        for p,g in zip([w1,b1,w2,b2,w3,b3],grads):
            p.assign_sub(lr*g)

        if step % 100 ==0:
            print(f"epoch:{epoch},step:{step},loss:{float(loss)}")
            losses.append(float(loss))
def test(epoch,test_db):
    total, total_correct = 0., 0.
    for x, y in test_db:
        # layer1.
        h1 = x @ w1 + b1
        h1 = tf.nn.relu(h1)
        # layer2
        h2 = h1 @ w2 + b2
        h2 = tf.nn.relu(h2)
        # output
        out = h2 @ w3 + b3
        # [b, 10] => [b]
        pre = tf.argmax(out,axis=1)
        y = tf.argmax(y,axis=1)
        corret = tf.equal(pre,y)
        total_correct += tf.reduce_sum(tf.cast(corret,dtype=tf.int32)).numpy()
        total += x.shape[0]

    print(epoch, 'Evaluate Acc:', total_correct / total)

    accs.append(total_correct / total)


def plot_loss(losses):
    plt.figure()
    plt.plot(losses,color='c0',marker='s',label='训练')
    plt.xlabel("epoch")
    plt.legend()
    plt.ylabel("loss")
    plt.show()

def main():
    train_db ,test_db = getDataLoader()
    EPOCHS = 100
    for epoch in range(EPOCHS):
        train(epoch)
        test(test_db)
if __name__ == '__main__':
    main()