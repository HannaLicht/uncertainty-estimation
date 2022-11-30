# https://lars76.github.io/2020/08/07/metrics-for-uncertainty-estimation.html

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions


# Expected Calibration Error (ECE)
def expected_calibration_error(y_true, y_pred, certainties=None, num_bins=15, two_returns=False):
    """
    :param y_true: indexs of correct labels (not one-hot vectors)
    :param y_pred:
    :param uncertainties:
    :param num_bins:
    :param two_returns:
    :return:
    """
    pred_y = np.argmax(y_pred, axis=-1).astype(np.float32)
    correct = (pred_y == y_true)
    if certainties is not None:
        prob_y = certainties
    else:
        prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    if two_returns:
        x, y = [], []
        for b in range(num_bins):
            mask = bins == b
            if np.any(mask):
                x.append(np.sum(prob_y[mask])/len(prob_y[mask]))
                y.append(np.sum(correct[mask])/len(correct[mask]))
        return x, y

    else:
        o = 0
        for b in range(num_bins):
            mask = bins == b
            if np.any(mask):
                o += np.abs(np.sum(correct[mask] - prob_y[mask]))
        return o / y_pred.shape[0]


def static_calibration_error(y_true, y_pred, num_bins=15):
    classes = y_pred.shape[-1]

    o = 0
    for cur_class in range(classes):
        correct = (cur_class == y_true).astype(np.float32)
        prob_y = y_pred[..., cur_class]

        b = np.linspace(start=0, stop=1.0, num=num_bins)
        bins = np.digitize(prob_y, bins=b, right=True)

        for b in range(num_bins):
            mask = bins == b
            if np.any(mask):
                o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / (y_pred.shape[0] * classes)


def reliability_diagram(y_true, output, certainties=None, num_bins=15, method=""):
    x, y = expected_calibration_error(y_true, output, certainties=certainties, num_bins=num_bins, two_returns=True)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(x, y, "s-", label=method)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.title("Reliability Diagram")


def uncertainty_diagram(y_true, y_pred, uncertainties, method=""):
    y_pred = np.argmax(y_pred, axis=-1).astype(np.float32)
    correct = (y_pred == y_true)

    b = np.linspace(start=tf.reduce_min(uncertainties), stop=tf.reduce_max(uncertainties), num=10)
    bins = np.digitize(uncertainties, bins=b, right=True)

    x, y = [], []
    for b in range(10):
        mask = bins == b
        if np.any(mask):
            x.append(np.sum(uncertainties[mask]) / len(uncertainties[mask]))
            y.append(np.sum(correct[mask]) / len(correct[mask]))
    plt.plot(x, y, "s-")
    plt.xlabel("Uncertainty")
    plt.ylim((-0.05, 1.05))
    plt.ylabel("Accuracy")
    plt.title(method)