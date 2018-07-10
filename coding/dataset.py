#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# dataset.py
#
# Created by Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>

import numpy as np
import matplotlib.pyplot as plt


def dataset2(num = 40):
    omega = np.random.randn(1)
    noise = 0.8 * np.random.randn(1, num)
    x = np.random.randn(2, num)
    y = (2 * (x[0, :] + x[1, :] + noise > 0) - 1)
    return(x, y)


if __name__ == '__main__':
    (x, y) = dataset2()
    for a, b in zip(x.T, y.T):
        marker = "o" if b == 1 else "x"
        color = "red" if b == 1 else "blue"
        plt.scatter(a[0], a[1], marker=marker, color=color)
    plt.show()
