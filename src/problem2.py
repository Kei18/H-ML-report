#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# problem2.py
#
# Created by Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>


import os
import numpy as np
import matplotlib.pyplot as plt
import lasso
import solver


# for visualization
def make_contour(A, mu, _lambda, step=0.05, xmin=-2, xmax=3.1, ymin=-2, ymax=3.1):
    x = np.arange(xmin, xmax, step)
    y = np.arange(ymin, ymax, step)
    loss = np.zeros((len(x), len(y)))
    for i in np.arange(len(x)):
        for j in np.arange(len(y)):
            w = np.array([x[i], y[j]])
            loss[j][i] = (lasso.loss_quadratic(w, A, mu, _lambda))
    return (x, y, loss)


if __name__ == '__main__':
    # params
    repeat_num = 50
    A = np.array([[3, 0.5], [0.5, 1]])
    mu = np.array([1, 2])
    w_init = np.array([3, -1])

    # optimal values from lecture slides
    w_opt_2 = solver.get_quadratic_opt(A, mu, 2)
    w_opt_4 = solver.get_quadratic_opt(A, mu, 4)
    w_opt_6 = solver.get_quadratic_opt(A, mu, 6)

    # implement lasso with proximal gradient
    w_2 = lasso.lasso(lasso.proximal_gradient, w_init, A, mu, 2, repeat_num)
    w_4 = lasso.lasso(lasso.proximal_gradient, w_init, A, mu, 4, repeat_num)
    w_6 = lasso.lasso(lasso.proximal_gradient, w_init, A, mu, 6, repeat_num)

    # implement lasso with acccelerated proximal gradient
    w_a_2 = lasso.lasso(lasso.accelerated_proximal_gradient,
                        w_init, A, mu, 2, repeat_num)
    w_a_4 = lasso.lasso(lasso.accelerated_proximal_gradient,
                        w_init, A, mu, 4, repeat_num)
    w_a_6 = lasso.lasso(lasso.accelerated_proximal_gradient,
                        w_init, A, mu, 6, repeat_num)

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
    # implement adagrad and comparing with PG and APG
    # parameter
    repeat_num = 500
    A = np.array([[250., 15.], [15., 4.]])
    mu = np.array([1, 2])
    _lambda = 0.89
    w_init = np.array([3, -1])

    # get optimal value
    w_opt = solver.get_quadratic_opt(A, mu, _lambda)

    # implement lasso
    w_pg = lasso.lasso(lasso.proximal_gradient, w_init, A, mu, _lambda, repeat_num)
    w_apg = lasso.lasso(lasso.accelerated_proximal_gradient, w_init,
                        A, mu, _lambda, repeat_num)
    w_adagrad = lasso.lasso(lasso.adagrad, w_init, A, mu, _lambda, repeat_num)
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
