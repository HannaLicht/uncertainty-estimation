# https://lars76.github.io/2020/08/07/metrics-for-uncertainty-estimation.html
from scipy.interpolate import interp1d
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


def get_normalized_certainties(pred_val, y_val, uncertainties_val, uncertainties_test, num_bins=10):
    y_pred = tf.argmax(pred_val, axis=-1)
    correct = (y_pred == tf.argmax(y_val, axis=-1))
    x = uncertainties_val
    y = [1. if c else 0. for c in correct]

    _, normalized_certainties = isotonic_regression(x, y, uncertainties_test)
    return tf.cast(normalized_certainties, tf.float32)


def reliability_diagram(y_true, output, certainties=None, num_bins=15, method=None, label_perfectly_calibrated=True,
                        color=None):
    x, y = expected_calibration_error(y_true, tf.argmax(output, axis=-1), certainties=certainties, num_bins=num_bins,
                                      two_returns=True)
    plt.plot(x, y, "s-", label=method, color=color)
    plt.plot([0, 1], [0, 1], "k:", label="Perfekt kalibriert" if label_perfectly_calibrated else None)
    plt.xlabel("Certainty")
    plt.ylabel("Accuracy")
    if method is not None or label_perfectly_calibrated:
        plt.legend(bbox_to_anchor=(0.5, 0.4))
    #plt.title("Calibration Plot")


def uncertainty_diagram(y_true, y_pred, uncertainties, title="", label=None, color=None):
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
    plt.plot(x, y, "s-", color=color, label=label, zorder=1)
    plt.title(title)
    plt.xlabel("Uncertainty Estimates")
    plt.ylim((-0.05, 1.05))
    plt.ylabel("Accuracy")

    if label is not None:
        plt.legend(bbox_to_anchor=(0.5, 1))


def plot_regression(y_true, y_pred, uncertainties, title="", label=False):
    y_pred = np.argmax(y_pred, axis=-1).astype(np.float32)
    correct = (y_pred == y_true)

    plt.title(title)
    plt.xlabel("Uncertainty Estimates")
    plt.ylim((-0.05, 1.05))
    plt.ylabel("Accuracy")

    y = [1. if c else 0. for c in correct]
    x = np.linspace(start=tf.reduce_min(uncertainties), stop=tf.reduce_max(uncertainties), num=100)
    regressor, normalized_certainties = isotonic_regression(uncertainties, y, x)
    plt.plot(x, normalized_certainties, color="black", label="Regressionsfunktion" if label else None, zorder=0)

    if label:
        plt.legend(bbox_to_anchor=(0.5, 1))

    return regressor


def isotonic_regression(x, y, uncertainties):
    regressor = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=False, out_of_bounds='clip')
    regressor = regressor.fit(x, y)
    return regressor, regressor.predict(uncertainties)