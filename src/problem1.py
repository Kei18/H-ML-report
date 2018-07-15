#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# problem1.py
#
# Created by Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>


import os
import numpy as np
import matplotlib.pyplot as plt
import dataset as db
import logistic


if __name__ == '__main__':
    # params of learning
    _eta = 0.1
    _lambda = 0.1
    repeat_num = 50
    data_num = 100
    w_init = np.zeros(2)

    # create training data
    (x, y) = db.dataset2(data_num)

    # implement logistic regression
    w_s = logistic.regression(logistic.direction_steepest, w_init,
                              x, y, _lambda, _eta, repeat_num)
    w_n = logistic.regression(logistic.direction_newton, w_init,
                              x, y, _lambda, _eta, repeat_num)

    # compute loss
    loss_s = [logistic.loss_func(w, x, y, _lambda) for w in w_s]
    loss_n = [logistic.loss_func(w, x, y, _lambda) for w in w_n]

    # plot training data
    plt.figure(figsize=(4, 4))
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
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.legend()
    filename = os.path.join("figs", "p1_logistic-regression_result.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()

    # plot loss
    plt.plot(np.arange(0, len(loss_s)), loss_s,
             color="green", label="steepest gradient method")
    plt.plot(np.arange(0, len(loss_n)), loss_n,
             color="orange", label="newton method")
    plt.xlim(0, repeat_num)
    plt.xlabel("t")
    plt.ylabel("J(w^t)")
    plt.legend()
    filename = os.path.join("figs", "p1_logistic-regression_loss.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()
