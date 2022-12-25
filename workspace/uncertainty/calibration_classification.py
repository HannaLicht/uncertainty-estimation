# https://lars76.github.io/2020/08/07/metrics-for-uncertainty-estimation.html

import numpy as np
from sklearn.isotonic import IsotonicRegression
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions


# Expected Calibration Error (ECE)
def expected_calibration_error(y_true, y_pred, certainties, num_bins=15, two_returns=False):
    """
    :param y_true: indexs of correct labels (not one-hot vectors)
    :param y_pred:
    :param uncertainties:
    :param num_bins:
    :param two_returns:
    :return:
    """
    pred_y = tf.argmax(y_pred, axis=-1)
    correct = tf.cast((pred_y == y_true), dtype=tf.float32)
    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(certainties, bins=b, right=True)

    if two_returns:
        x, y = [], []
        for b in range(num_bins):
            mask = bins == b
            if np.any(mask):
                x.append(np.sum(certainties[mask])/len(certainties[mask]))
                y.append(np.sum(correct[mask])/len(correct[mask]))
        return x, y

    else:
        o = 0
        for b in range(num_bins):
            mask = bins == b
            if np.any(mask):
                o += tf.abs(tf.reduce_sum(correct[mask]) - tf.reduce_sum(certainties[mask]))
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


def get_normalized_certainties(pred_val, y_val, uncertainties_val, uncertainties_test):
    y_pred = tf.argmax(pred_val, axis=-1)
    correct = (y_pred == tf.argmax(y_val, axis=-1))

    b = np.linspace(start=tf.reduce_min(uncertainties_val), stop=tf.reduce_max(uncertainties_val), num=10)
    bins = np.digitize(uncertainties_val, bins=b, right=True)

    x, y = [], []
    for b in range(10):
        mask = bins == b
        if np.any(mask):
            x.append(np.sum(uncertainties_val[mask]) / len(uncertainties_val[mask]))
            y.append(np.sum(correct[mask]) / len(correct[mask]))

    normalized_certainties = isotonic_regression(x, y, bins, uncertainties_test)
    return normalized_certainties


def reliability_diagram(y_true, output, certainties=None, num_bins=15, method=None, label_perfectly_calibrated=True):
    x, y = expected_calibration_error(y_true, output, certainties=certainties, num_bins=num_bins, two_returns=True)
    plt.plot(x, y, "s-", label=method)
    plt.plot([0, 1], [0, 1], "k:", label="Perfekt kalibriert" if label_perfectly_calibrated else None)
    plt.xlabel("Konfidenz")
    plt.ylabel("Accuracy")
    if method is not None or label_perfectly_calibrated:
        plt.legend(loc="upper left")
    #plt.title("Calibration Plot")


def uncertainty_diagram(y_true, y_pred, uncertainties, method="", label=""):
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
    plt.plot(x, y, "s-", label=label)
    plt.xlabel("Uncertainty")
    plt.ylim((-0.05, 1.05))
    plt.ylabel("Accuracy")
    plt.title(method)

    if label == "Validierungsdaten":
        normalized_certainties = isotonic_regression(x, y, bins, x)
        plt.plot(x, normalized_certainties, "k:", label="Isotonische Regression Validierung")

    plt.legend(loc="lower left")


def isotonic_regression(x, y, bins, uncertainties):
    weights = [list(bins).count(i) for i in range(10)]
    regressor = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=False, out_of_bounds='clip')
    regressor = regressor.fit(x, y, sample_weight=weights)
    return regressor.predict(uncertainties)