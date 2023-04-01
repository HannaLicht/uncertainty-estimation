# https://lars76.github.io/2020/08/07/metrics-for-uncertainty-estimation.html
from scipy.interpolate import interp1d
import numpy as np
from sklearn.isotonic import IsotonicRegression
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions

"""
notation of the functions:
y_true - correct label (index of correct class, not one-hot)
y_pred - index of predicted class
outputs - softmax output
"""


# Expected Calibration Error (ECE)
def expected_calibration_error(y_true, y_pred, certainties, num_bins=15, two_returns=False):
    correct = tf.cast((y_pred == y_true), dtype=tf.float32)
    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(certainties, bins=b, right=True)

    if two_returns:
        x, y = [], []
        for b in range(num_bins):
            mask = bins == b
            if np.any(mask):
                x.append(tf.reduce_sum(certainties[mask])/len(certainties[mask]))
                y.append(tf.reduce_sum(correct[mask])/len(correct[mask]))
        return x, y

    else:
        o = 0
        for b in range(num_bins):
            mask = bins == b
            if np.any(mask):
                o += tf.abs(tf.reduce_sum(correct[mask]) - tf.reduce_sum(certainties[mask]))

        return o / len(y_true)


def get_normalized_certainties(output, y_val, uncertainties_val, uncertainties_test, num_bins=10):
    y_pred = tf.argmax(output, axis=-1)
    correct = (y_pred == tf.argmax(y_val, axis=-1))
    x = uncertainties_val
    y = [1. if c else 0. for c in correct]

    _, normalized_certainties = isotonic_regression(x, y, uncertainties_test)
    return tf.cast(normalized_certainties, tf.float32)


def reliability_diagram(y_true, outputs, certainties=None, num_bins=15, method=None, label_perfectly_calibrated=True,
                        color=None):
    x, y = expected_calibration_error(y_true, tf.argmax(outputs, axis=-1), certainties=certainties, num_bins=num_bins,
                                      two_returns=True)
    plt.plot(x, y, "s-", label=method, color=color)
    plt.plot([0, 1], [0, 1], "k:", label="Perfekt kalibriert" if label_perfectly_calibrated else None)
    plt.xlabel("Certainty")
    plt.ylabel("Accuracy")
    if method is not None or label_perfectly_calibrated:
        plt.legend(loc="lower right")


def uncertainty_diagram(y_true, outputs, uncertainties, title="", label=None, color=None):
    y_pred = np.argmax(outputs, axis=-1).astype(np.float32)
    correct = (y_pred == y_true)

    b = np.linspace(start=tf.reduce_min(uncertainties), stop=tf.reduce_max(uncertainties), num=8)
    bins = np.digitize(uncertainties, bins=b, right=True)

    x, y = [], []
    for b in range(8):
        mask = bins == b
        if np.any(mask):
            x.append(np.sum(uncertainties[mask]) / len(uncertainties[mask]))
            y.append(np.sum(correct[mask]) / len(correct[mask]))
    plt.plot(x, y, "s-", color=color, label=label, zorder=1, linewidth=1.)
    plt.title(title)
    plt.xlabel("Uncertainty Estimates")
    plt.ylim((-0.05, 1.05))
    plt.ylabel("Accuracy")

    if label is not None:
        plt.legend(bbox_to_anchor=(0.5, 1))


def plot_regression(y_true, outputs, uncertainties, title="", label=False, text="", style="-", utest=None):
    y_pred = np.argmax(outputs, axis=-1).astype(np.float32)
    correct = (y_pred == y_true)

    plt.title(title)
    plt.xlabel("Uncertainty Estimates")
    plt.ylim((-0.05, 1.05))
    plt.ylabel("Accuracy")

    y = [1. if c else 0. for c in correct]
    if utest is None:
        x = np.linspace(start=tf.reduce_min(uncertainties), stop=tf.reduce_max(uncertainties), num=100)
    else:
        x = np.linspace(start=tf.reduce_min(utest), stop=tf.reduce_max(utest), num=100)

    regressor, normalized_certainties = isotonic_regression(uncertainties, y, x)
    plt.plot(x, normalized_certainties, color="black", label="G " + text if label else None, zorder=0, linewidth=1.,
             linestyle=style)

    if label:
        plt.legend(bbox_to_anchor=(0.5, 1))

    return regressor


def isotonic_regression(x, y, uncertainties):
    regressor = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=False, out_of_bounds='clip')
    regressor = regressor.fit(x, y)
    return regressor, regressor.predict(uncertainties)