#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# solver.py
#
# Created by Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>


import cvxopt
import numpy as np


def get_opt(A, mu, _lambda):
    P = cvxopt.matrix(A)
    q = cvxopt.matrix(_lambda/2*np.ones(mu.shape[0]) - np.dot(mu.T, A))
    sol = cvxopt.solvers.qp(P, q)
    return np.array(sol['x']).flatten()


if __name__ == '__main__':
    A = np.array([[3, 0.5], [0.5, 1]])
    mu = np.array([1, 2])
    _lambda = 2
    print(get_opt(A, mu, _lambda))
