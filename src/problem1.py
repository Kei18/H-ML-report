#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# problem1.py
#
# Created by Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>


import os
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import dataset as db


def loss_func(w, x, y, _lambda):
    return np.sum(np.log(1 + np.exp(-np.dot(w.T, x) * y)), axis=1) + _lambda * np.dot(w.T, w)

def loss_grad(w, x, y, _lambda):
    n = x.shape[1]
    ans = np.sum((- y * x) / (1 + np.exp(np.dot(w.T, x) * y)), axis=1) / n + 2 * _lambda * w
    return ans

def loss_hessian(w, x, y, _lambda):
    n = x.shape[1]
    tmp1 = (np.exp(-np.dot(w.T, x) * y) / (1 + np.exp(-np.dot(w.T, x) * y))**2).flatten()
    tmp2 = np.array([np.dot(_x.reshape(2,1), _x.reshape(1,2)) for _x in x.T])
    tmp3 = np.sum(np.array([a * b for (a, b) in zip(tmp1, tmp2)]), axis=0) / n
    return tmp3 + 2 * _lambda * np.eye(w.shape[0], w.shape[0])

def direction_steepest(w, x, y, _lambda):
    return - loss_grad(w, x, y, _lambda)

def direction_newton(w, x, y, _lambda):
    hessian = loss_hessian(w, x, y, _lambda)
    return - np.dot(np.linalg.inv(hessian), loss_grad(w, x, y, _lambda))

def logistic_regression(func_direction, x, y, repeat_num=100, _step=0.1, _lambda=0.1):
    # initialize weight
    w = np.zeros(x.shape[0])
    # history
    w_hist = [w]
    # update weight
    for i in range(0, repeat_num):
        w = w + _step * func_direction(w, x, y, _lambda)
        w_hist.append(w)
        cnt_correct = np.sum(2 * (np.dot(w.T, x) > 0) - 1 == y)
        print("itr:%d" % i, "correct:%d/%d" % (cnt_correct, y.shape[1]))
    return w_hist


if __name__ == '__main__':
    plt.figure(figsize=(4, 4))

    # params of learning
    _step = 0.1
    _lambda = 0.1
    repeat_num = 50
    data_num = 100

    # create training data
    (x, y) = db.dataset2(data_num)

    w_s = logistic_regression(direction_steepest, x, y, repeat_num, _step, _lambda)
    w_n = logistic_regression(direction_newton, x, y, repeat_num, _step, _lambda)
    loss_s = [loss_func(w, x, y, _lambda) for w in w_s]
    loss_n = [loss_func(w, x, y, _lambda) for w in w_n]

    # plot training data
    for a, b in zip(x.T, y.T):
        marker = "o" if b == 1 else "x"
        color = "red" if b == 1 else "blue"
        plt.scatter(a[0], a[1], marker=marker, color=color)
    # plot classification, w_1*x_1 + w_2*x_2 = 0
    x_plot = np.arange(-2, 2.1, 0.1)
    y_s = - w_s[-1][0] / w_s[-1][1] * x_plot
    y_n = - w_n[-1][0] / w_n[-1][1] * x_plot
    plt.plot(x_plot, y_s, color="green", label="steepest gradient method")
    plt.plot(x_plot, y_n, color="orange", label="newton method")

    # for file name
    now = dt.now().strftime("%Y-%m-%d-%H-%M-%S")

    # save learning result
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.legend()
    filename = os.path.join("figs", "p1_logistic-regression_result-%s.pdf" % now)
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()

    # save learning process
    plt.plot(np.arange(0, len(loss_s)), loss_s, color="green", label="steepest gradient method")
    plt.plot(np.arange(0, len(loss_n)), loss_n, color="orange", label="newton method")
    plt.xlim(0, repeat_num)
    plt.xlabel("t")
    plt.ylabel("J(w^t)")
    plt.legend()
    filename = os.path.join("figs", "p1_logistic-regression_loss-%s.pdf" % now)
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()
