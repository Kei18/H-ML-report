#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# lasso.py
# utils of lasso
#
# Created by Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>

import numpy as np


# soft threshold func
def soft_threshold(q, mu):
    if mu > q:
        return mu - q
    elif mu < -q:
        return mu + q
    else:
        return 0

# soft thresholding func for vector
def soft_threshold_array(q, arr):
    return np.array([soft_threshold(q, mu) for mu in arr])

# loss of quadratic form with L2 norm
def loss_quadratic(w, A, mu, _lambda):
    return np.dot(np.dot((w - mu).T, A), w - mu) + _lambda * np.sum(np.abs(w))

# gradient of quadratic form
def grad_quadratic(w, A, mu):
    return 2 * np.dot(A, (w - mu))

# scalar func of accerelated proximal gradient
def q_scalar_func(t):
    return float(t - 1) / float(t + 2)

# update with proximal gradient
def proximal_gradient(w_hist, A, mu, _lambda, _gamma, t, g_hist):
    _q = _lambda / _gamma
    w = soft_threshold_array(_q, w_hist[-1] - grad_quadratic(w_hist[-1], A, mu) / _gamma)
    return w

# update with accelerated proximal gradient
def accelerated_proximal_gradient(w_hist, A, mu, _lambda, _gamma, t, g_hist):
    if t <= 1:
        return proximal_gradient(w_hist, A, mu, _lambda, _gamma, t, g_hist)
    _q = _lambda / _gamma
    v = w_hist[-1] + q_scalar_func(t) * (w_hist[-1] - w_hist[-2])
    w = soft_threshold_array(_q, v - grad_quadratic(v, A, mu) / _gamma)
    return w

# update with adagarad, g_hist means history of gradient
def adagrad(w_hist, A, mu, _lambda, _gamma, t, g_hist):
    _q = _lambda / _gamma
    _epsilon=0.02
    _eta = 500 / _gamma
    # there are more simple implementations
    # Here, the equations are truly depicted
    G = np.diag(np.sum(np.array(g_hist)**2, axis=0))
    H = np.sqrt(G) + _epsilon * np.eye(w_hist[-1].shape[0])
    arg_arr = w_hist[-1] - _eta * np.dot(np.linalg.inv(H), g_hist[-1])
    w = np.array([soft_threshold(_q, _mu) for _q, _mu
                  in zip(_eta * _lambda / np.diag(H), arg_arr)])
    g_hist.append(grad_quadratic(w, A, mu))
    return w

# implement lasso
def lasso(func, w_init, A, mu, _lambda, repeat_num=50):
    # obtain eigen value of 2A, then, use as the inverse of the lerning rate
    (eigs, vec) = np.linalg.eig(2*A)
    _gamma = np.max(eigs)
    # history of params
    w_hist = [w_init]
    # for adagrad
    g_hist = [grad_quadratic(w_hist[-1], A, mu)]
    # update params
    for t in range(0, repeat_num):
        w = func(w_hist, A, mu, _lambda, _gamma, t, g_hist)
        w_hist.append(w)
    return w_hist
