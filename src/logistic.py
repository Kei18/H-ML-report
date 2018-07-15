#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# logistic.py
# utils of logistic regression
#
# Created by Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>

import numpy as np


# logistic loss with L2 norm
def loss_func(w, x, y, _lambda):
    return np.sum(np.log(1 + np.exp(-np.dot(w.T, x) * y)), axis=1) + _lambda * np.dot(w.T, w)

# gradient of logistic loss with L2 norm
def grad(w, x, y, _lambda):
    n = x.shape[1]
    ans = np.sum((- y * x) / (1 + np.exp(np.dot(w.T, x) * y)), axis=1) / n + 2 * _lambda * w
    return ans

# hessian of logistic loss with L2 norm
def hessian(w, x, y, _lambda):
    n = x.shape[1]
    tmp1 = (np.exp(-np.dot(w.T, x) * y) / (1 + np.exp(-np.dot(w.T, x) * y))**2).flatten()
    tmp2 = np.array([np.dot(_x.reshape(2,1), _x.reshape(1,2)) for _x in x.T])
    tmp3 = np.sum(np.array([a * b for (a, b) in zip(tmp1, tmp2)]), axis=0) / n
    return tmp3 + 2 * _lambda * np.eye(w.shape[0], w.shape[0])

# get params of steepest gradient method
def direction_steepest(w, x, y, _lambda):
    return - grad(w, x, y, _lambda)

# get params of newton method
def direction_newton(w, x, y, _lambda):
    h = hessian(w, x, y, _lambda)
    return - np.dot(np.linalg.inv(h), grad(w, x, y, _lambda))

# implement logistic regresssion
def regression(func, w_init, x, y, _lambda=0.1, _eta=0.1, repeat_num=100):
    # history of params
    w_hist = [w_init]
    # update params
    for t in range(0, repeat_num):
        w = w_hist[-1] + _eta * func(w_hist[-1], x, y, _lambda)
        w_hist.append(w)
    return w_hist
