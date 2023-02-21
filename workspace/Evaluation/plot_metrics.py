import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from functions import adjust_lightness, COLORS
import tensorflow_probability as tfp
tfd = tfp.distributions

MODEL_NAME = "CNN_cifar100"

methods = ["MCD SE", "Bag SE", "DA SE", "Soft SE", "MCD MI", "Bag MI", "DA MI", "NUC Tr", "NUC Va"]
if MODEL_NAME == "CNN_cifar10_100":
    del methods[-1]
colors = [COLORS[m] for m in methods]


def decor_plot(ax):
    plt.ylim(-0.02, 1.1)
    plt.xlim(-0.02, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.arrow(x=0, y=0, dx=1.05, dy=0, head_width=0.02, head_length=0.06, fc='black', ec='black', lw=0.2)
    ax.arrow(x=0, y=0, dx=0, dy=1.05, head_width=0.02, head_length=0.06, fc='black', ec='black', lw=0.2)


def plot_roc_and_pr_curves(results_metrics, vari, labels):
    plt.figure(figsize=(9, 3.5))
    #rec_good, pr_good = tf.concat((rec_good, [0.]), axis=0), tf.concat((pr_good, [1.]), axis=0)

    ax = plt.subplot(1, 2, 1)
    plt.title("ROC-Kurven")
    for (pre, spe, rec), (pre_var, spe_var, rec_var), c in zip(results_metrics, vari, colors):
        plt.plot(1-spe, rec, color=c, linewidth=1., zorder=1)
        plt.fill_between(1-spe, rec + rec_var, rec-rec_var, color=c, alpha=0.1, zorder=0)
    plt.plot([0, 1], [0, 1], "--", color="black", linewidth=1.)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    decor_plot(ax)

    ax = plt.subplot(1, 2, 2)
    plt.title("PR-Kurven")
    for (pre, spe, rec), (pre_var, spe_var, rec_var), lbl, c in zip(results_metrics, vari,  labels, colors):
        plt.plot(rec[:-1], pre[:-1], label=lbl, color=c, linewidth=1., zorder=1)
        plt.fill_between(rec[:-1], pre[:-1] + pre_var[:-1], pre[:-1] - pre_var[:-1], color=c, alpha=0.1, zorder=0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    with open('../Results/accuracies.json') as json_file:
        data = json.load(json_file)
    y = data[MODEL_NAME]["test"]
    plt.plot([0, 1], [1-y, 1-y], "--", color="black", linewidth=1.)

    decor_plot(ax)

    plt.legend(bbox_to_anchor=(1.5, 1.0))
    plt.subplots_adjust(left=0.09, right=0.83, bottom=0.16, top=0.9, wspace=0.2, hspace=0.35)
    plt.savefig("../plots/roc_and_pr_curve_" + MODEL_NAME + ".pdf")
    plt.show()


def subplot_evaluation(values, v, eval_metric: str, method: str, c):
    thr = list(np.linspace(0.6, 1, 20))
    while values[0] is None:
        values = values[1:]
        thr = thr[1:]

    l, = plt.plot(thr, values, label=method, color=c, linewidth=1., zorder=1)
    plt.fill_between(thr, values + v, values - v, color=c, alpha=0.1, zorder=0)

    plt.xlabel("Certainty-Schwellenwert")
    plt.ylabel(eval_metric)
    plt.xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    plt.grid(visible=True, color="gainsboro", linestyle='dashed', zorder=0)

    return l


def plot_metric(results_metrics, vari, labels, index_metric, indices_best, y_min, y_max):
    plt.figure(figsize=(10, 3.3))
    if index_metric == 0:
        name = "Precision"
        loc = "lower left"
    elif index_metric == 1:
        name = "Specificity"
        loc = "lower left"
    else:
        name = "Recall"
        loc = "lower right"

    ax = plt.subplot(1, 3, 1)
    plt.title("Shannon Entropie")
    plt.ylim(y_min, y_max)
    ax.set_axisbelow(True)
    for res, v, lbl, c in zip(results_metrics[:4], vari[:4], labels[:4], colors[:4]):
        subplot_evaluation(res[index_metric], v[index_metric], name, lbl, c)
    plt.legend(loc=loc)

    ax = plt.subplot(1, 3, 2)
    plt.title("Mutual Information & NUC")
    plt.ylim(y_min, y_max)
    ax.set_axisbelow(True)
    for res, v, lbl, c in zip(results_metrics[4:], vari[4:], labels[4:], colors[4:]):
        subplot_evaluation(res[index_metric], v[index_metric], name, lbl, c)
    plt.legend(loc=loc)

    ax = plt.subplot(1, 3, 3)
    plt.title("beste SE, MI und NUC-Methode")
    plt.ylim(y_min, y_max)
    ax.set_axisbelow(True)
    best_results = [results_metrics[i] for i in indices_best]
    best_vars = [vari[i] for i in indices_best]
    best_colors = [colors[i] for i in indices_best]
    best_lbl = [labels[i] for i in indices_best]
    for res, v, lbl, c in zip(best_results, best_vars, best_lbl, best_colors):
        subplot_evaluation(res[index_metric], v[index_metric], name, lbl, c)
    plt.legend(loc=loc)

    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.9, wspace=0.3, hspace=0.35)
    plt.savefig("../plots/" + name + "_" + MODEL_NAME + ".pdf")
    plt.show()


def plot_pre_spe_rec(results_metrics, vari, labels):
    if MODEL_NAME == "CNN_cifar10":
        indices_pre = [1, 3, 5, 7]
        indices_spe = [0, 3, 4, 7]
        indices_rec = [1, 3, 5, 8]
        min_pre, max_pre = 0.255, 0.64
        min_spe, max_spe = -0.05, 0.92
        min_rec, max_rec = 0.39, 1.02

    elif MODEL_NAME == "effnetb3":
        indices_pre = [0, 1, 4, 5]
        indices_spe = [0, 2, 4, 6]
        indices_rec = [1, 5, 7]
        min_pre, max_pre = 0.34, 0.68
        min_spe, max_spe = -0.04, 0.84
        min_rec, max_rec = 0.63, 1.02

    elif MODEL_NAME == "CNN_cifar100":
        indices_pre = [0, 3, 5, 7]
        indices_spe = [2, 3, 6, 7]
        indices_rec = [1, 3, 4, 8]
        min_pre, max_pre = 0.48, 0.77
        min_spe, max_spe = -0.19, 0.65
        min_rec, max_rec = 0.856, 1.005

    else:
        quit()

    plot_metric(results_metrics, vari, labels, 0, indices_pre, min_pre, max_pre)
    plot_metric(results_metrics, vari, labels, 1, indices_spe, min_spe, max_spe)
    plot_metric(results_metrics, vari, labels, 2, indices_rec, min_rec, max_rec)


with open('../Results/pre_spe_rec.json') as json_file:
    data = json.load(json_file)

results = []
variances = []
for m in methods:
    pre = tf.reduce_mean(tf.convert_to_tensor(data[MODEL_NAME][m]["pre"]["curves"]), axis=0)
    spe = tf.reduce_mean(tf.convert_to_tensor(data[MODEL_NAME][m]["spe"]["curves"]), axis=0)
    rec = tf.reduce_mean(tf.convert_to_tensor(data[MODEL_NAME][m]["rec"]["curves"]), axis=0)
    pre_var = tf.math.reduce_std(tf.convert_to_tensor(data[MODEL_NAME][m]["pre"]["curves"]), axis=0)
    spe_var = tf.math.reduce_std(tf.convert_to_tensor(data[MODEL_NAME][m]["spe"]["curves"]), axis=0)
    rec_var = tf.math.reduce_std(tf.convert_to_tensor(data[MODEL_NAME][m]["rec"]["curves"]), axis=0)
    results.append([pre, spe, rec])
    variances.append([pre_var, spe_var, rec_var])

plot_roc_and_pr_curves(results, variances, methods)

if MODEL_NAME == "effnetb3":
    methods.pop(-2)
    colors.pop(-2)

results = []
variances = []
for m in methods:
    pre = tf.reduce_mean(tf.convert_to_tensor(data[MODEL_NAME][m]["pre"]["calibrated"]), axis=0)
    spe = tf.reduce_mean(tf.convert_to_tensor(data[MODEL_NAME][m]["spe"]["calibrated"]), axis=0)
    rec = tf.reduce_mean(tf.convert_to_tensor(data[MODEL_NAME][m]["rec"]["calibrated"]), axis=0)
    pre_var = tf.math.reduce_std(tf.convert_to_tensor(data[MODEL_NAME][m]["pre"]["calibrated"]), axis=0)
    spe_var = tf.math.reduce_std(tf.convert_to_tensor(data[MODEL_NAME][m]["spe"]["calibrated"]), axis=0)
    rec_var = tf.math.reduce_std(tf.convert_to_tensor(data[MODEL_NAME][m]["rec"]["calibrated"]), axis=0)
    results.append([pre, spe, rec])
    variances.append([pre_var, spe_var, rec_var])

plot_pre_spe_rec(results, variances, methods)

