#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# svm.py
# utils of support vector machine
#
# Created by Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>

import numpy as np
import itertools
from lasso import soft_threshold_array

# hinge loss with L1 norm
def hinge_loss(x, y, w, _lambda):
    tmp = np.vstack([np.zeros(x.shape[1]), np.ones(x.shape[1]) - np.dot(w.T, x) * y])
    return np.sum(np.max(tmp, axis=0), axis=0) + _lambda * np.dot(w.T, w)

# obtain dual lagrange func
def dual_lagrange(alpha, K, _lambda):
    return - np.dot(alpha.T, np.dot(K, alpha)) / (4 * _lambda) + np.sum(alpha)

# obtain w from alpha
def get_w_value(x, y, _lambda, alpha):
    return np.sum((alpha * y).flatten() * x, axis=1) / (2 * _lambda)

# projection for projected gradient
def projection(i, s, l):
    if i > l:
        return l
    elif i < s:
        return s
    else:
        return i

# vector projection for projected gradient
def vec_projection(vec, s, l):
    return np.array([projection(i, _s, _l) for (i, _s, _l) in zip(vec, s, l)])

# compute K
def get_K(x, y):
    n = x.shape[1]
    K = np.zeros([n, n])
    for i, j in itertools.product(range(n), range(n)):
        K[i][j] = y[:,i][0] * y[:,j][0] * np.dot(x[:,i].T, x[:,j])
    return K

# subgradient of hinge loss func
def subgradient_hinge(x, y, w, theta=0.5):
    if y * np.dot(w, x) > 1:
        return np.zeros(x.shape[0])
    elif y * np.dot(w, x) < 1:
        return - y * x
    else:
        return - theta * y * x

# proximal subgradient method
def proximal_subgradient(w_init, x, y, _lambda, _eta, repeat_num=50):
    # history of params
    w_hist = [w_init]
    # update params
    for t in range(0, repeat_num):
        # get subgradient
        g = np.sum(np.array([subgradient_hinge(_x, _y, w_hist[-1])
                             for (_x, _y) in zip(x.T, y.T)]), axis=0)
        w = soft_threshold_array(_eta * _lambda, w_hist[-1] - _eta * g)
        w_hist.append(w)
    return w_hist

# support vector machine
def svm(alpha_init, x, y, _lambda, _eta, repeat_num=50):
    n = x.shape[1]
    # compute K
    K = get_K(x, y)
    # initialize param
    alpha_hist = [alpha_init]
    # update param
    for t in range(0, repeat_num):
        tmp = alpha_hist[-1] - _eta * (np.dot(K, alpha_hist[-1])
                                       / (2 * _lambda) - np.ones(n))
        alpha = vec_projection(tmp, np.zeros(n), np.ones(n))
        alpha_hist.append(alpha)
    return alpha_hist
