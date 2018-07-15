#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# problem3.py
#
# Created by Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>


import numpy as np
import matplotlib.pyplot as plt
import os
import dataset as db
import svm


if __name__ == '__main__':
    # params
    _lambda = 10
    _eta = 0.1
    repeat_num = 50
    data_num = 100
    alpha_init = np.zeros(data_num)

    # create trainig data
    (x, y) = db.dataset2(data_num)

    # implement svm
    alpha = svm.svm(alpha_init, x, y, _lambda, _eta, repeat_num)

    # compute params
    w = np.array([svm.get_w_value(x, y, _lambda, _alpha) for _alpha in alpha])

    # compute score of dual of lagrange
    n = x.shape[1]
    K = svm.get_K(x, y)
    duals = [svm.dual_lagrange(_alpha, K, _lambda) for _alpha in alpha]

    # compute score of hinge function
    loss = [svm.hinge_loss(x, y, _w, _lambda) for _w in w]

    # plot training data
    plt.figure(figsize=(4, 4))
    for a, b in zip(x.T, y.T):
        marker = "o" if b == 1 else "x"
        color = "red" if b == 1 else "blue"
        plt.scatter(a[0], a[1], marker=marker, color=color)
    # plot classification, w_1*x_1 + w_2*x_2
    x_plot = np.arange(-2, 2.1, 0.1)
    y_0 = - w[-1][0] / w[-1][1] * x_plot
    y_1 = (- w[-1][0] * x_plot + 1) / w[-1][1]
    y_m1 = (- w[-1][0] * x_plot - 1) / w[-1][1]
    plt.plot(x_plot, y_m1, color="c", label="dot(w,x)=-1")
    plt.plot(x_plot, y_0, color="m", label="dot(w,x)=0")
    plt.plot(x_plot, y_1, color="y", label="dot(w,x)=1")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.legend()
    filename = os.path.join("figs", "p3_svm_result.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()

    # plot score of duality gap
    plt.figure(figsize=(4, 4))
    plt.plot(np.arange(0, repeat_num + 1), duals, color="#014757", label="dual Lagrange")
    plt.plot(np.arange(0, repeat_num + 1), loss, color="#B11500", label="hinge loss")
    plt.xlim(0, repeat_num)
    plt.ylim(0, 100)
    plt.ylabel("score of func")
    plt.xlabel("t")
    plt.legend()
    filename = os.path.join("figs", "p3_svm_gap.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()
