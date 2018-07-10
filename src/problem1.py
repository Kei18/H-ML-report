#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# problem1.py
#
# Created by Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>


import numpy as np
import matplotlib.pyplot as plt
import dataset as db


def loss_func(w, x, y, _lambda):
    return np.sum(np.log(1 + np.exp(-np.dot(w.T, x) * y)), axis=1) + _lambda * np.dot(w.T, w)

def logistic_grad(w, x, y, _lambda):
    n = x.shape[1]
    ans = np.sum((- y * x) / (1 + np.exp(np.dot(w.T, x) * y)), axis=1) / n + 2 * _lambda * w
    return ans

def logistic_steepest(x, y, repeat_num=1000, _step=0.1, _lambda=0.1):
    # initialize weight
    w = np.zeros(x.shape[0])
    # update weight
    for i in range(0, repeat_num):
        w = w - _step * logistic_grad(w, x, y, _lambda)
        loss = loss_func(w, x, y, _lambda)
        cnt_correct = np.sum(2 * (np.dot(w.T, x) > 0) - 1 == y)
        print("itr:%d" % i, "loss-func %f" % loss, "correct:%d/%d" % (cnt_correct, y.shape[1]))
    return w

if __name__ == '__main__':
    plt.figure(figsize=(4, 4))

    # params of learning
    _step = 0.1
    _lambda = 0.1
    repeat_num = 100
    data_num = 100

    # create training data
    (x, y) = db.dataset2(data_num)

    # learning by steepest gradient method
    w = logistic_steepest(x, y, repeat_num, _step, _lambda)

    # plot training data
    for a, b in zip(x.T, y.T):
        marker = "o" if b == 1 else "x"
        color = "red" if b == 1 else "blue"
        plt.scatter(a[0], a[1], marker=marker, color=color)
    # plot classification, w_1*x_1 + w_2*x_2 = 0
    steepest_x = np.arange(-2, 2.1, 0.1)
    steepest_y = - w[0] / w[1] * steepest_x
    plt.plot(steepest_x, steepest_y, color="green", label="seepest gradient method")

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.legend()
    plt.show()
