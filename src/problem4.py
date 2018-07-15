#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# problem4.py
#
# Created by Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>

import matplotlib.pyplot as plt
import numpy as np
import os
import dataset as db
import svm
import solver


if __name__ == '__main__':
    # params
    _eta = 0.05
    _lambda = 1
    repeat_num = 50
    data_num = 100
    w_init = np.zeros(2)

    # create training data
    (x, y) = db.dataset2(data_num)

    # get opt
    (xi, e, w_opt) = solver.get_opt_lpboost(x, y, _lambda)
    loss_opt = svm.hinge_loss(x, y, w_opt, _lambda)

    # implement proximal subgradient method
    w = svm.proximal_subgradient(w_init, x, y, _lambda, _eta, repeat_num)
    loss = [svm.hinge_loss(x, y, _w, _lambda) for _w in w]

    # plot training data
    plt.figure(figsize=(4, 4))
    for a, b in zip(x.T, y.T):
        marker = "o" if b == 1 else "x"
        color = "red" if b == 1 else "blue"
        plt.scatter(a[0], a[1], marker=marker, color=color)
    # plot classification, w_1*x_1 + w_2*x_2 = 0
    x_plot = np.arange(-2, 2.1, 0.1)
    y_plot = - w[-1][0] / w[-1][1] * x_plot
    y_opt = - w_opt[0] / w_opt[1] * x_plot

    # plot result
    plt.plot(x_plot, y_plot, color="green", label="proximal subgradient")
    plt.plot(x_plot, y_opt, color="orange", label="optimal")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.legend()
    filename = os.path.join("figs", "p4_proximal-subgradient_result.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()

    # plot loss
    plt.figure(figsize=(4, 4))
    plt.plot(np.arange(0, repeat_num+1), loss,
             color="green", label="proximal subgradient")
    plt.plot(np.arange(0, repeat_num+1), np.ones(repeat_num+1) * loss_opt,
             color="orange", label="optimal")
    plt.xlim(0, repeat_num)
    plt.xlabel("t")
    plt.ylabel("score of hinge loss")
    plt.legend()
    filename = os.path.join("figs", "p4_proximal-subgradient_loss.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()
