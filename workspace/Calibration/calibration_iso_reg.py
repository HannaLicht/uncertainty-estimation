import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

sys.path.append("/home/urz/hlichten")
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.Ensemble import BaggingEns, DataAugmentationEns, RandomInitShuffleEns, ENSEMBLE_LOCATION
from uncertainty.calibration_classification import reliability_diagram, uncertainty_diagram, expected_calibration_error, \
    plot_regression
from functions import get_data, CNN

method = "ensemble"
metric = "MI"
isontonic_reg = True


fig = plt.figure(figsize=(12, 3.5)) if isontonic_reg else plt.figure(figsize=(10, 3))

for count, (model_name, title) in enumerate(zip(["CNN_cifar10_1000", "CNN_cifar10", "CNN_cifar100"],
                                                ["CNN Cifar10 (1000 Bilder)", "CNN Cifar10 (gesamt)", "CNN Cifar100 (gesamt)"])):

    xtrain, ytrain, xval, yval, xtest, ytest, cl, _, _ = get_data("cifar10" if count != 2 else "cifar100",
                                                                  num_data=1000 if count == 0 else None)
    model = CNN(classes=cl)
    model.load_weights("../models/classification/" + model_name + "/cp.ckpt")
    ax = plt.subplot(1, 3, count + 1)
    ax.set_axisbelow(True)
    plt.grid(visible=True, color="gainsboro", linestyle='dashed', zorder=0)

    path = ENSEMBLE_LOCATION + "/" + method + "/" + model_name

    if method == "mc_drop":
        estimator = [MCDropoutEstimator(model, xtest, cl, xval=xval, yval=yval)]
    elif method == "bagging":
        estimator = [BaggingEns(xtest, cl, path, X_val=xval, y_val=yval, val=True)]
    elif method == "data_augmentation":
        estimator = [DataAugmentationEns(xtest, cl, path, X_val=xval, y_val=yval, val=True)]
    elif method == "rand_initialization_shuffle":
        estimator = [RandomInitShuffleEns(xtest, cl, path, X_val=xval, y_val=yval, val=True)]
    colors = [["darkblue"], ["yellowgreen"]]
    labels = [""]
    styles = ["-"]
    if method == "ensemble":
        estimator = [BaggingEns(xtest, cl, ENSEMBLE_LOCATION + "/bagging/" + model_name, X_val=xval, y_val=yval,
                                val=True),
                     DataAugmentationEns(xtest, cl, ENSEMBLE_LOCATION + "/data_augmentation/" + model_name,
                                         X_val=xval, y_val=yval, val=True)]
        colors = [["darkblue", "royalblue"], ["green", "yellowgreen"]]
        labels = ["Bagging", "Data Aug."]
        styles = ["-", "--"]

    if isontonic_reg:

        if metric == "SE":
            for est, color, label, style in zip(estimator, colors[0], labels, styles):
                regressor = plot_regression(tf.argmax(yval, axis=-1), est.val_p_ens,
                                            est.uncertainties_shannon_entropy(val=True), style=style,
                                            label=True if model_name == "CNN_cifar100" else False, text=label)
                lbl = "Testprädiktionen Bag." if label == "Bagging" else "Testprädiktionen DA"
                uncertainty_diagram(tf.argmax(ytest, axis=-1), est.p_ens,
                                    est.uncertainties_shannon_entropy(), title=title, color=color,
                                    label=lbl if model_name == "CNN_cifar100" else None)
            plt.xlabel("Shannon Entropie")
            if model_name == "CNN_cifar10_1000":
                start = 0
                stop = 2.2
                step = 0.5
            elif model_name == "CNN_cifar10":
                start = 0
                stop = 2.2
                step = 0.5
            else:
                start = 0
                stop = 4.1
                step = 1

            '''plt.plot([2, 2], [0, regressor.predict([2])[0]], color="gray", linestyle='dashed', zorder=2)
            plt.plot([start, 2], [regressor.predict([2])[0], regressor.predict([2])[0]], color="gray",
                     linestyle='dashed', zorder=2)
            plt.scatter(2, regressor.predict([2])[0], c="black", marker='o', zorder=10)
'''
        else:
            for est, color, label, style in zip(estimator, colors[1], labels, styles):
                lbl = "Testprädiktionen Bag." if label == "Bagging" else "Testprädiktionen DA"
                regressor = plot_regression(tf.argmax(yval, axis=-1), est.val_p_ens,
                                            est.uncertainties_mutual_information(val=True), style=style, text=label,
                                            label=True if model_name == "CNN_cifar100" else False)
                uncertainty_diagram(tf.argmax(ytest, axis=-1), est.p_ens,
                                    est.uncertainties_mutual_information(), title=title, color=color,
                                    label=lbl if model_name == "CNN_cifar100" else None)
            plt.xlabel("Mutual Information")

            if model_name == "CNN_cifar10_1000":
                start = 0
                stop = 0.75
                step = 0.2
            elif model_name == "CNN_cifar10":
                start = 0
                stop = 0.65
                step = 0.2
            else:
                start = 0
                stop = 1.1
                step = 0.2

        plt.xticks(np.arange(start, stop, step=step))

    else:
        plt.title(title)
        estimator = estimator[0]
        reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=estimator.p_ens,
                            certainties=estimator.normalized_certainties_shannon_entropy(),
                            label_perfectly_calibrated=False, num_bins=10,
                            method="SE" if model_name == "CNN_cifar100" else None)
        reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=estimator.p_ens, num_bins=10,
                            certainties=estimator.normalized_certainties_mutual_information(),
                            label_perfectly_calibrated=False, color="yellowgreen",
                            method="MI" if model_name == "CNN_cifar100" else None)

        ece_se = expected_calibration_error(tf.argmax(ytest, axis=-1), estimator.get_ensemble_prediction(),
                                            estimator.normalized_certainties_shannon_entropy()).numpy()
        ece_mi = expected_calibration_error(tf.argmax(ytest, axis=-1), estimator.get_ensemble_prediction(),
                                            estimator.normalized_certainties_mutual_information()).numpy()
        plt.text(0.02, 0.95, "ECE SE: {:.3f}".format(ece_se), color="brown", weight="bold")
        plt.text(0.02, 0.87, "ECE MI: {:.3f}".format(ece_mi), color="brown", weight="bold")

if isontonic_reg:
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plot_name = '../plots/calibration_' + method + "_" + metric + "_isotonic_regression.png"
    plt.subplots_adjust(left=0.05, right=0.81, bottom=0.16, top=0.9, wspace=0.27, hspace=0.35)
else:
    plt.legend(loc="lower right")
    plt.subplots_adjust(left=0.06, right=0.96, bottom=0.16, top=0.9, wspace=0.3, hspace=0.35)
    plot_name = '../plots/calibration_' + method + ".png"

plt.savefig(plot_name, dpi=300)
plt.show()