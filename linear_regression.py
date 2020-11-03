# -*- coding: utf-8 -*-
# @Time    : 2020-10-22 15:17
# @Author  : RichardoMu
# @File    : linear_regression.py
# @Software: PyCharm
import numpy as np

def get_data():

    data = []
    for i in range(100):
        x = np.random.uniform(-10.,10.)
        eps = np.random.normal(0,0.1)
        y = 1.723 * x + 21.124 + eps
        data.append([x,y])
    return np.array(data)


def mse(data,w_current ,b_current ):
    loss = 0.0
    for i, [x,y] in enumerate(data):
        loss += (w_current * x + b_current - y) ** 2
    loss /= len(data)
    return loss


def step_gradient(data,b_current ,w_current,lr):
    b_gradient = 0.0
    w_gradient = 0.0
    data_len = float(len(data))
    for i, [x,y] in enumerate(data):
        b_gradient += 2/data_len * (w_current * x + b_current - y)
        w_gradient += 2/data_len * (w_current * x + b_current - y) * x
    b_new = b_current - lr * b_gradient
    w_new = w_current - lr * w_gradient
    return [b_new,w_new]


def gradient_descent(data,w_starting,b_starting ,lr,num_iterator):
    b = b_starting
    w = w_starting
    for i in range(num_iterator):
        b , w = step_gradient(data,b,w,lr)
        loss = mse(data,w,b)
        if i % 100 == 0:
            print(f"loss:{loss},w_current :{w},b_current:{b}")
    return [b,w]


def main():
    np.random.seed(1234)
    b_starting,w_starting = 0.0, 0.0
    num_iterator = 1000
    lr = 0.01
    data = get_data()
    b,w = gradient_descent(data,w_starting,b_starting,lr,num_iterator)
    loss = mse(data,w,b)
    print(f"loss:{loss},w_current :{w},b_current:{b}")


if __name__ == '__main__':
    main()
