#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# problem2.py
#
# Created by Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>


import os
import numpy as np
import matplotlib.pyplot as plt
import solver

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

def q_scalar_func(t):
    return float(t - 1) / float(t + 2)

def lasso(A, mu, _lambda, repeat_num=50, rule=None,
          epsilon=0.02, w_init=np.array([3, -1])):
    # obtain eigen value of 2A, then, use as the inverse of the lerning rate
    (eigs, vec) = np.linalg.eig(2*A)
    _gamma = np.max(eigs)
    q = _lambda / _gamma
    # initialize weight
    w = w_init
    # history
    w_hist = [w]
    # for adagrad
    g_hist = [phi_grad(w, A, mu)]
    eta = 500 / _gamma
    # update weight
    for t in range(0, repeat_num):
        if rule == "accelerated" and t > 1:
            # accelerated proximal gradient update
            v = w_hist[-1] + q_scalar_func(t) * (w_hist[-1] - w_hist[-2])
            w = soft_threshold_array(q, v - phi_grad(v, A, mu) / _gamma)
        elif rule == "adagrad":
            # there are more simple implementations
            # Here, the equations are truly depicted
            G = np.diag(np.sum(np.array(g_hist)**2, axis=0))
            H = np.sqrt(G) + epsilon * np.eye(w.shape[0])
            arg_arr = w - eta * np.dot(np.linalg.inv(H), g_hist[-1])
            w = np.array([soft_threshold(_q, _mu) for _q, _mu
                          in zip(eta * _lambda / np.diag(H), arg_arr)])
            g_hist.append(phi_grad(w, A, mu))
        else:
            # proximal gradient update
            w = soft_threshold_array(q, w - phi_grad(w, A, mu) / _gamma)
        w_hist.append(w)
        print("itr:%d" % t, w)
    return w_hist

# for visualization
def make_contour(A, mu, _lambda, step=0.05, xmin=-2, xmax=3.1, ymin=-2, ymax=3.1):
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
    w_opt_2 = solver.get_opt(A, mu, 2)
    w_opt_4 = solver.get_opt(A, mu, 4)
    w_opt_6 = solver.get_opt(A, mu, 6)

    # implement lasso with proximal gradient
    w_2 = lasso(A, mu, 2, repeat_num)
    w_4 = lasso(A, mu, 4, repeat_num)
    w_6 = lasso(A, mu, 6, repeat_num)

    # implement lasso with acccelerated proximal gradient
    w_a_2 = lasso(A, mu, 2, repeat_num, "accelerated")
    w_a_4 = lasso(A, mu, 4, repeat_num, "accelerated")
    w_a_6 = lasso(A, mu, 6, repeat_num, "accelerated")

    # compute dist from opt
    dist_2 = [np.sum(np.abs(w - w_opt_2)) for w in w_2]
    dist_4 = [np.sum(np.abs(w - w_opt_4)) for w in w_4]
    dist_6 = [np.sum(np.abs(w - w_opt_6)) for w in w_6]
    dist_a_2 = [np.sum(np.abs(w - w_opt_2)) for w in w_a_2]
    dist_a_4 = [np.sum(np.abs(w - w_opt_4)) for w in w_a_4]
    dist_a_6 = [np.sum(np.abs(w - w_opt_6)) for w in w_a_6]

    # compute contour to plot data
    (x, y, loss_2) = make_contour(A, mu, 2)
    (x, y, loss_4) = make_contour(A, mu, 4)
    (x, y, loss_6) = make_contour(A, mu, 6)

    # plot result
    def plt_config_result():
        plt.figure(figsize=(5, 4))
        plt.xlim(-1.5, 3)
        plt.ylim(-1.5, 3)
        plt.xticks([-1, 0, 1, 2, 3])
        plt.yticks([-1, 0, 1, 2, 3])
        plt.xlabel("w_1")
        plt.ylabel("w_2")

    # lambda=2
    plt_config_result()
    plt.contour(x, y, loss_2, 50)
    plt.colorbar()
    plt.plot([_w[0] for _w in w_2], [_w[1] for _w in w_2], label="PG",
             color="red", marker="o", ms=2, linewidth=0.8)
    plt.plot([_w[0] for _w in w_a_2], [_w[1] for _w in w_a_2], label="APG",
             color="blue", marker="x", ms=2, linewidth=0.8)
    plt.legend()
    filename = os.path.join("figs", "p2_lasso_result_lambda-2.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()

    # lambda=4
    plt_config_result()
    plt.contour(x, y, loss_4, 50)
    plt.colorbar()
    plt.plot([_w[0] for _w in w_4], [_w[1] for _w in w_4], label="PG",
             color="red", marker="o", ms=2, linewidth=0.8)
    plt.plot([_w[0] for _w in w_a_4], [_w[1] for _w in w_a_4], label="APG",
             color="blue", marker="x", ms=2, linewidth=0.8)
    plt.legend()
    filename = os.path.join("figs", "p2_lasso_result_lambda-4.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()

    # lambda=6
    plt_config_result()
    plt.contour(x, y, loss_6, 50)
    plt.colorbar()
    plt.plot([_w[0] for _w in w_6], [_w[1] for _w in w_6], label="PG",
             color="red", marker="o", ms=2, linewidth=0.8)
    plt.plot([_w[0] for _w in w_a_6], [_w[1] for _w in w_a_6], label="APG",
             color="blue", marker="x", ms=2, linewidth=0.8)
    plt.legend()
    filename = os.path.join("figs", "p2_lasso_result_lambda-6.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()

    # plot dist
    def plt_config_result_dist():
        plt.figure(figsize=(4, 4))
        plt.xlim(0, repeat_num)
        plt.yscale('log')
        plt.ylim(1e-08, 1e+2)
        plt.xlabel("t")
        plt.ylabel("||w^(t) - w_opt||")

    # lambda=2
    plt_config_result_dist()
    plt.plot(np.arange(repeat_num + 1), dist_2, color="red", label="PG")
    plt.plot(np.arange(repeat_num + 1), dist_a_2, color="blue", label="APG")
    plt.legend()
    filename = os.path.join("figs", "p2_lasso_dist_lambda-2.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()

    # lambda=4
    plt_config_result_dist()
    plt.plot(np.arange(repeat_num + 1), dist_4, color="red", label="PG")
    plt.plot(np.arange(repeat_num + 1), dist_a_4, color="blue", label="APG")
    plt.legend()
    filename = os.path.join("figs", "p2_lasso_dist_lambda-4.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()

    # lambda=6
    plt_config_result_dist()
    plt.plot(np.arange(repeat_num + 1), dist_6, color="red", label="PG")
    plt.plot(np.arange(repeat_num + 1), dist_a_6, color="blue", label="APG")
    plt.legend()
    filename = os.path.join("figs", "p2_lasso_dist_lambda-6.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()

    #----------------------------------------------------
    # parameter
    A = np.array([[250., 15.], [15., 4.]])
    mu = np.array([1, 2])
    _lambda = 0.89
    repeat_num = 500

    # get optimal value
    w_opt = solver.get_opt(A, mu, _lambda)

    # implement lasso
    w_pg = lasso(A, mu, _lambda, repeat_num)
    w_apg = lasso(A, mu, _lambda, repeat_num, "accelerated")
    w_adagrad = lasso(A, mu, _lambda, repeat_num, "adagrad")
    (x, y, loss) = make_contour(A, mu, _lambda)

    # compute distance to opt
    dist_pg = [np.sum(np.abs(w - w_opt)) for w in w_pg]
    dist_apg = [np.sum(np.abs(w - w_opt)) for w in w_apg]
    dist_adagrad = [np.sum(np.abs(w - w_opt)) for w in w_adagrad]

    # plot result
    plt_config_result()
    plt.contour(x, y, loss, 50)
    plt.colorbar()
    plt.plot([_w[0] for _w in w_pg], [_w[1] for _w in w_pg], label="PG",
             color="red", marker="o", ms=2, linewidth=0.2)
    plt.plot([_w[0] for _w in w_apg], [_w[1] for _w in w_apg], label="APG",
             color="blue", marker="x", ms=2, linewidth=0.8)
    plt.plot([_w[0] for _w in w_adagrad], [_w[1] for _w in w_adagrad], label="AdaGrad",
             color="green", marker="s", ms=2, linewidth=0.8)
    plt.legend()
    filename = os.path.join("figs", "p2_lasso_result_pg-apg-adagrad.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')
    plt.clf()

    # plot dist
    plt.figure(figsize=(4, 4))
    plt.plot(np.arange(repeat_num + 1), dist_pg, color="red", label="PG")
    plt.plot(np.arange(repeat_num + 1), dist_apg, color="blue", label="APG")
    plt.plot(np.arange(repeat_num + 1), dist_adagrad, color="green", label="AdaGrad")
    plt.xlim(0, repeat_num)
    plt.yscale('log')
    plt.xlabel("t")
    plt.ylabel("||w^(t) - w_opt||")
    plt.legend()
    filename = os.path.join("figs", "p2_lasso_dist_adagrad.pdf")
    plt.savefig(filename, pad_inches=0.05, transparent=True, bbox_inches='tight')

