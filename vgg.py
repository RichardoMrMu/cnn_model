# -*- coding: utf-8 -*-
# @Time    : 2020-11-02 16:39
# @Author  : RichardoMu
# @File    : vgg.py
# @Software: PyCharm

import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def preprocess(x,y):
    # x = tf.expand_dims(x,axis=2)
    x = tf.image.resize(x,[224,224])
    x = tf.cast(x,dtype=tf.float32)/255.
    print(f"y.shape:{y.shape}")
    y = tf.cast(y,dtype=tf.int32)
    y = tf.one_hot(y,depth=10)
    y = tf.squeeze(y,axis=0)

    return tf.expand_dims(x,axis=0),tf.expand_dims(y,axis=0)

class VGG(tf.keras.Model):
    def __init__(self):
        super(VGG, self).__init__()
        self.layer1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation=tf.keras.layers.LeakyReLU()),
            tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),

        ])
        self.layer2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128,(3,3,),padding='same',activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),
        ])
        self.layer3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256,(3,3),padding='same',activation='relu'),
            tf.keras.layers.Conv2D(256, (3, 3),padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),
        ])
        self.layer4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(512,(3,3),padding='same',activation='relu'),
            tf.keras.layers.Conv2D(512,(3,3),padding='same',activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),
        ])
        self.layer4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu'),
            tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        ])
        self.dense1 = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096,activation='relu'),
            tf.keras.layers.Dense(4096,activation='relu'),
            tf.keras.layers.Dense(1000,activation='relu'),
            tf.keras.layers.Dense(10,activation=None)
        ])
        # self.ConLayer = tf.keras.Sequential([
        #     self.layer1(),
        #     self.layer2(),
        #     self.layer3(),
        #     self.layer4()
        # ])
        # self.DenseLayer = tf.keras.Sequential([
        #     self.dense1()
        # ])


    def call(self,input_x):
        # print(input_x.shape)
        x = self.layer1(input_x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = self.dense1(x)
        # print(x.shape)
        return x

class VGG13(tf.keras.Model):
    def __init__(self):
        super(VGG13, self).__init__()
configs = {
    'A': [
        '3-64','M',
        '3-128','M',
        '3-256','3-256','M',
        '3-512','3-512','M',
        '3-512','3-512','M'
    ],
    'B': [
        '3-64','3-64','M',
        '3-128','3-128','M',
        '3-256','3-256','M',
        '3-512','3-512','M',
        '3-512','3-512','M'
    ],
    'C': [
        '3-64','3-64','M',
        '3-128','3-128','M',
        '3-256','3-256','1-256','M',
        '3-512', '3-512','1-512', 'M',
        '3-512', '3-512', '1-512', 'M',
    ],
    'D': [
        '3-64','3-64','M',
        '3-128','3-128','M',
        '3-256','3-256','3-256','M',
        '3-512', '3-512','3-512', 'M',
        '3-512', '3-512', '3-512', 'M',
    ] ,
    'E': [
        '3-64','3-64','M',
        '3-128','3-128','M',
        '3-256','3-256','3-256','3-256','M',
        '3-512', '3-512','3-512','3-512', 'M',
        '3-512', '3-512', '3-512', '3-512','M',
    ] ,

}

class VGGNet(tf.keras.Model):
    """ VGGNet Architecture"""
    def __init__(self,config,class_count,rate=0.4):
        super(VGGNet, self).__init__()
        self.config = config
        self.class_count = class_count
        self.rate = rate
        self.conv_net = self.get_conv_net()
        self.fc_net = self.get_fc_net()



    def get_conv_net(self):
        """
        return the convolutional layers of the network
        :return:
        """
        layers = []
        print(configs[self.config])
        for layer in configs[self.config]:
            if layer == "M":
                layers.append(tf.keras.layers.MaxPool2D((2,2),strides=2,padding='same'))
            else:
                layer = layer.split('-')
                kernel_size = int(layer[0])
                filters = int(layer[1])
                layers.append(tf.keras.layers.Conv2D(filters=filters,
                                                     kernel_size=kernel_size,
                                                     padding='same',kernel_initializer='he_normal'))
                layers.append(tf.keras.layers.BatchNormalization())
                layers.append(tf.keras.layers.LeakyReLU())
        return tf.keras.Sequential(layers)


    def get_fc_net(self):
        """
        :return:the fully connected layers of the network
        """
        layers = []
        layers.append(tf.keras.layers.Flatten())
        layers.append(tf.keras.layers.LeakyReLU())
        layers.append(tf.keras.layers.Dropout(rate=self.rate))
        layers.append(tf.keras.layers.Dense(4096,kernel_initializer='he_normal'))
        layers.append(tf.keras.layers.Dense(4096,kernel_initializer='he_normal'))
        layers.append(tf.keras.layers.Dense(1000,kernel_initializer='he_normal'))
        layers.append(tf.keras.layers.Dense(self.class_count,kernel_initializer='he_normal'))
        return tf.keras.Sequential(layers)


    # def call(self,input_x,training=None):
    #     x = self.conv_net(input_x,training=training)
    #     x = self.fc_net(x,training=training)
    #     return x
    def call(self,input_x):
        x = self.conv_net(input_x)
        x = self.fc_net(x)
        return x


def main():
    lr=0.001
    tf.random.set_seed(1234)
    (x,y),(x_val,y_val) = tf.keras.datasets.cifar10.load_data()

    x1,y1 = x[0],y[0]
    x2,y2 = x[1],y[1]
    x3,y3 = x[2],y[2]
    x1 ,y1 = preprocess(x1,y1)
    x2 ,y2 = preprocess(x2,y2)
    x3, y3 = preprocess(x3,y3)
    print(x1.shape,y1.shape)
    model = VGGNet('A',10)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(x=x1,y=y1,epochs=1)
    model.evaluate(x=x2, y=y2)
    model.fit(x=x2, y=y2, epochs=1)
    model.save("model")
    # output = model(input_x,training=True)
    # output2 = model(input_x)
    # output1 = model(input_x,training=False)
    # print(output)
    # print(output2)
    # print(output1)
    # model.evaluate(x=input_x,y=y)
    # print(output.shape)
    # for layer in model.layers:
    #     print(layer.summary())

if __name__ == '__main__':
    main()