#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# solver.py
# obtain optimal solution using cvxopt
#
# Created by Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>

import cvxopt
import numpy as np


# obtain optimal for (w-\mu)^T A (w-\mu)
def get_quadratic_opt(A, mu, _lambda):
    P = cvxopt.matrix(A)
    q = cvxopt.matrix(_lambda/2*np.ones(mu.shape[0]) - np.dot(mu.T, A))
    sol = cvxopt.solvers.qp(P, q)
    return np.array(sol['x']).flatten()

# obtain optimal for lpboost
def get_opt_lpboost(x, y, _lambda):
    d = x.shape[0]
    n = x.shape[1]
    A1 = np.hstack([np.diag(-1 * np.ones(n)),
                    np.zeros([n, d]),
                    - (y * x).T])
    A2 = np.hstack([np.diag(-1 * np.ones(n)),
                    np.zeros([n, d]),
                    np.zeros([n, d])])
    A3 = np.hstack([np.zeros([d, n]),
                    np.diag(-1 * np.ones(d)),
                    np.diag(-1 * np.ones(d))])
    A4 = np.hstack([np.zeros([d, n]),
                    np.diag(-1 * np.ones(d)),
                    np.diag(np.ones(d))])
    A5 = np.hstack([np.zeros([d, n]),
                    np.diag(-1 * np.ones(d)),
                    np.zeros([d, d])])
    A = cvxopt.matrix(np.vstack([A1, A2, A3, A4, A5]))
    c = cvxopt.matrix(np.hstack([np.ones(n), _lambda * np.ones(d), np.zeros(d)]))
    b = cvxopt.matrix(np.hstack([-1 * np.ones(n),
                                 np.zeros(n),
                                 np.zeros(d),
                                 np.zeros(d),
                                 np.zeros(d)]))
    sol = cvxopt.solvers.qp(cvxopt.matrix(np.zeros([n+2*d, n+2*d])),
                            c, G=A, h=b)
    z = np.array(sol['x']).flatten()
    xi = z[0:n]
    e = z[n:n+d]
    w = z[-d:]
    return (xi, e, w)


if __name__ == '__main__':
    A = np.array([[3, 0.5], [0.5, 1]])
    mu = np.array([1, 2])
    _lambda = 2
    print(get_opt(A, mu, _lambda))
