# -*- coding: utf-8 -*-
# @Time    : 2020-11-02 8:56
# @Author  : RichardoMu
# @File    : main.py
# @Software: PyCharm

import tensorflow as tf
from alexnet import AlexNet
import os
from vgg import VGGNet
from googlenet import GoogleNet
from resnet import resnet_50
from senet import se_resnet_50
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus :
    tf.config.experimental.set_memory_growth(gpu,True)
# image classification
batchsz = 128
lr = 1e-2
EPOCH = 20
best_acc = 0.
tf.random.set_seed(1234)
# model = AlexNet()
# model = VGGNet('A',10)
# model = GoogleNet()
# model = resnet_50()
model = se_resnet_50()
def preprocess(x,y):
    # x = tf.expand_dims(x,axis=2)
    x = tf.image.resize(x,[224,224])
    x = tf.cast(x,dtype=tf.float32)/255.
    print(f"y.shape:{y.shape}")
    y = tf.cast(y,dtype=tf.int32)
    y = tf.one_hot(y,depth=10)
    y = tf.squeeze(y,axis=0)

    return x,y


# def train(model,epoch,train_db):
#     for step,(x,y) in enumerate(train_db):
#         with tf.GradientTape() as tape:
#             logits = model(x)
#             y_onehot = tf.one_hot(y,depth=10)
#             loss = lossFunction(y_onehot,logits)
#             loss = tf.reduce_mean(loss)
#         grads = tape.gradient(loss,model.trainable_variables)
#         optimizer.apply_gradients(grads,model.trainable_variables)
#         if step % 100 == 0:
#             print(f"epoch:{epoch},step:{step},loss:{loss}")
#
#
# def validation(model,epoch,test_db):
#     for step,(x,y) in enumerate(test_db):
#         logits = model(x)
#         y_onehot = tf.one_hot(y,depth=10)
#         loss = lossFunction(y_onehot,logits)
#         loss = tf.reduce_mean(loss)
#         prec = tf.equal(y_onehot,logits)
#         prec = tf.cast(prec,type=tf.int32)
#         prec = tf.reduce_sum(prec)/len(x)
#         if step % 50 == 0:
#             print(f"epoch:{epoch},step:{step},loss:{loss},precission:{prec}")
#         if ((epoch > 5) & (prec > best_acc)):
#             best_acc = prec
#             model.sa


def main():

    (x,y),(x_val,y_val) = tf.keras.datasets.cifar10.load_data()
    print(x.shape,y.shape,y[0])
    train_db = tf.data.Dataset.from_tensor_slices((x,y))
    train_db = train_db.shuffle(20000).map(preprocess).batch(batchsz)
    x ,y = next(iter(train_db))
    print(x.shape,y.shape)
    test_db = tf.data.Dataset.from_tensor_slices((x_val,y_val))
    test_db = test_db.map(preprocess).batch(batchsz)

    # model = AlexNet()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # train
    model.fit(train_db,epochs=20)
    scores = model.evaluate(test_db,verbose=1)
    print(f"final test loss and accuracy:",scores)
    model.save("model")
    model.save_weights("model1",save_format='tf')
if __name__ == '__main__':
    main()