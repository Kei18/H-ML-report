#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# problem2.py
#
# Created by Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>


import os
import numpy as np
import matplotlib.pyplot as plt
import dataset as db

def loss_func(w, A, mu, _lambda):
    return np.dot(np.dot((w - mu).T, A), w - mu) + _lambda * np.sum(w)

def soft_threshold_array(q, arr):
    return np.array([soft_threshold(q, mu) for mu in arr])

def soft_threshold(q, mu):
    if mu > q:
        return mu - q
    elif mu < -q:
        return mu + q
    else:
        return 0

def phi_grad(w, A, mu):
    return 2 * np.dot(A, (w - mu))

def lasso(A, mu, _lambda, repeat_num=50, w_init=np.array([3, -1])):
    # obtain eigen value of 2A, then, use as the inverse of the lerning rate
    (eigs, vec) = np.linalg.eig(2*A)
    _gamma = np.max(eigs)
    q = _lambda / _gamma
    # initialize weight
    w = w_init
    # history
    w_hist = [w]
    # update weight
    for i in range(0, repeat_num):
        w = soft_threshold_array(q, w - phi_grad(w, A, mu) / _gamma)
        w_hist.append(w)
        print("itr:%d" % i, w)
    return w_hist

# for visualization
def make_contour(w, A, mu, _lambda, step=0.05, xmin=-2, xmax=3.1, ymin=-2, ymax=3.1):
    x = np.arange(xmin, xmax, step)
    y = np.arange(ymin, ymax, step)
    loss = np.zeros((len(x), len(y)))
    for i in np.arange(len(x)):
        for j in np.arange(len(y)):
            w = np.array([x[i], y[j]])
            loss[j][i] = (loss_func(w, A, mu, _lambda))
    return (x, y, loss)


if __name__ == '__main__':
    # params
    repeat_num = 50
    A = np.array([[3, 0.5], [0.5, 1]])
    mu = np.array([1, 2])

    # optimal values from lecture slides
    w_opt_2 = np.array([0.82, 1.09])
    w_opt_4 = np.array([0.64, 0.18])
    w_opt_6 = np.array([0.33, 0])

    # implement lasso
    w_2 = lasso(A, mu, 2, repeat_num)
    w_4 = lasso(A, mu, 4, repeat_num)
    w_6 = lasso(A, mu, 6, repeat_num)

    # compute dist from opt
    dist_2 = [np.sum(np.abs(w - w_opt_2)) for w in w_2]
    dist_4 = [np.sum(np.abs(w - w_opt_4)) for w in w_4]
    dist_6 = [np.sum(np.abs(w - w_opt_6)) for w in w_6]

    # compute contour to plot data
    (x, y, loss_2) = make_contour(w, A, mu, 2)
    (x, y, loss_4) = make_contour(w, A, mu, 4)
    (x, y, loss_6) = make_contour(w, A, mu, 6)

    # plot result
    def plt_config_result():
        plt.figure(figsize=(5, 4))
        plt.xlim(-1.5, 3)
        plt.ylim(-1.5, 3)
        plt.xticks([-1, 0, 1, 2, 3])
        plt.yticks([-1, 0, 1, 2, 3])
        plt.xlabel("w_1")
        plt.ylabel("w_2")

    plt_config_result()
    plt.contour(x, y, loss_2, 50)
    plt.colorbar()
    plt.plot([_w[0] for _w in w_2], [_w[1] for _w in w_2],
             color="red", marker="o", ms=4, linewidth=1)
    filename = os.path.join("figs", "p2_lasso_result_lambda-2.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()

    plt_config_result()
    plt.contour(x, y, loss_4, 50)
    plt.colorbar()
    plt.plot([_w[0] for _w in w_4], [_w[1] for _w in w_4],
             color="red", marker="o", ms=4, linewidth=1)
    filename = os.path.join("figs", "p2_lasso_result_lambda-4.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()

    plt_config_result()
    plt.contour(x, y, loss_6, 50)
    plt.colorbar()
    plt.plot([_w[0] for _w in w_6], [_w[1] for _w in w_6],
             color="red", marker="o", ms=4, linewidth=1)
    filename = os.path.join("figs", "p2_lasso_result_lambda-6.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()

    # plot dist
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(repeat_num + 1), dist_2, label="lambda=2")
    plt.plot(np.arange(repeat_num + 1), dist_4, label="lambda=4")
    plt.plot(np.arange(repeat_num + 1), dist_6, label="lambda=6")
    plt.xlim(0, repeat_num)
    plt.yscale('log')
    plt.xlabel("t")
    plt.ylabel("||w^(t) - w_opt||")
    plt.legend()
    filename = os.path.join("figs", "p2_lasso_dist.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
