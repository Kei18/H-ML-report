#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# problem1.py
#
# Created by Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>


import numpy as np
import dataset as db


def logistic_grad(w, x, y, _lambda=1):
    n = w.shape[0]
    ans = np.sum((- y * x) / (1 + np.exp(np.dot(w.T, x) * y)), axis=1) / n + 2 * _lambda * w
    return ans

def logistic_learn(x, y, step=1, repeat_num=100):
    # initialize weight
    w = np.zeros(x.shape[0])
    for _ in range(0, repeat_num):
        w = w - step * logistic_grad(w, x, y)
        print(w, np.sum(2 * (np.dot(w.T, x) > 0) - 1 == 1))
    return w

if __name__ == '__main__':
    (x, y) = db.dataset2()
    w = logistic_learn(x, y)
