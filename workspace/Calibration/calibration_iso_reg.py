import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

sys.path.append("/home/urz/hlichten")
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.Ensemble import BaggingEns, DataAugmentationEns, RandomInitShuffleEns, ENSEMBLE_LOCATION
from uncertainty.calibration_classification import reliability_diagram, uncertainty_diagram, expected_calibration_error, \
    plot_regression
from functions import get_train_and_test_data, CNN

method = "mc_drop"
metric = "MI"
isontonic_reg = False


fig = plt.figure(figsize=(9, 2.8))

for count, (model_name, title) in enumerate(zip(["CNN_cifar10_100", "CNN_cifar10", "CNN_cifar100"],
                                                ["CNN Cifar10 (100 Bilder)", "CNN Cifar10 (gesamt)", "CNN Cifar100 (gesamt)"])):

    xtrain, ytrain, xval, yval, xtest, ytest, cl = get_train_and_test_data("cifar10" if count != 2 else "cifar100",
                                                                           validation_test_split=True)
    model = CNN(classes=cl)
    model.load_weights("../models/classification/" + model_name + "/cp.ckpt")
    plt.subplot(1, 3, count + 1)
    path = ENSEMBLE_LOCATION + "/" + method + "/" + model_name

    if method == "mc_drop":
        estimator = MCDropoutEstimator(model, xtest, cl, xval=xval, yval=yval)
    elif method == "bagging":
        estimator = BaggingEns(xtest, cl, path, X_val=xval, y_val=yval, val=True)
    elif method == "data_augmentation":
        estimator = DataAugmentationEns(xtest, cl, path, X_val=xval, y_val=yval, val=True)
    elif method == "rand_initialization_shuffle":
        estimator = RandomInitShuffleEns(xtest, cl, path, X_val=xval, y_val=yval, val=True)
    else:
        raise NotImplementedError

    if isontonic_reg:

        if metric == "SE":
            regressor = plot_regression(tf.argmax(yval, axis=-1), estimator.val_p_ens,
                                        estimator.uncertainties_shannon_entropy(val=True),
                                        label=True if model_name == "CNN_cifar100" else False)
            uncertainty_diagram(tf.argmax(ytest, axis=-1), estimator.p_ens,
                                estimator.uncertainties_shannon_entropy(), title=title,
                                label="Testdaten" if model_name == "CNN_cifar100" else None)
            plt.xlabel("Shannon Entropie")
            if model_name == "CNN_cifar10_100":
                start = 1.25
                stop = 2.35
                step = 0.25
            elif model_name == "CNN_cifar10":
                start = 0
                stop = 2.2
                step = 0.5
            else:
                start = 0
                stop = 4.1
                step = 1

            plt.plot([2, 2], [0, regressor.predict([2])[0]], color="gray", linestyle='dashed', zorder=2)
            plt.plot([start, 2], [regressor.predict([2])[0], regressor.predict([2])[0]], color="gray",
                     linestyle='dashed', zorder=2)
            plt.scatter(2, regressor.predict([2])[0], c="black", marker='o', zorder=10)

        else:
            regressor = plot_regression(tf.argmax(yval, axis=-1), estimator.val_p_ens,
                                        estimator.uncertainties_mutual_information(val=True),
                                        label=True if model_name == "CNN_cifar100" else False)
            uncertainty_diagram(tf.argmax(ytest, axis=-1), estimator.p_ens,
                                estimator.uncertainties_mutual_information(), title=title,
                                label="Testdaten" if model_name == "CNN_cifar100" else None, color="green")
            plt.xlabel("Mutual Information")

            if model_name == "CNN_cifar10_100":
                start = 0
                stop = 0.7
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
        reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=estimator.p_ens,
                            certainties=estimator.normalized_certainties_shannon_entropy(),
                            label_perfectly_calibrated=False, num_bins=10,
                            method="SE" if model_name == "CNN_cifar100" else None)
        reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=estimator.p_ens, num_bins=10,
                            certainties=estimator.normalized_certainties_mutual_information(),
                            label_perfectly_calibrated=False, color="green",
                            method="MI" if model_name == "CNN_cifar100" else None)

        ece_se = expected_calibration_error(tf.argmax(ytest, axis=-1), estimator.get_ensemble_prediction(),
                                            estimator.normalized_certainties_shannon_entropy()).numpy()
        ece_mi = expected_calibration_error(tf.argmax(ytest, axis=-1), estimator.get_ensemble_prediction(),
                                            estimator.normalized_certainties_mutual_information()).numpy()
        plt.text(0.02, 0.95, "ECE SE: {:.3f}".format(ece_se), color="brown", weight="bold")
        plt.text(0.02, 0.87, "ECE MI: {:.3f}".format(ece_mi), color="brown", weight="bold")
        if model_name == "CNN_cifar100":
            plt.legend(loc="lower right")

if isontonic_reg:
    plot_name = '../plots/calibration_' + method + "_" + metric + "_isotonic_regression.png"
    plt.subplots_adjust(left=0.06, right=0.88, bottom=0.16, top=0.9, wspace=0.3, hspace=0.35)
else:
    plt.subplots_adjust(left=0.06, right=0.96, bottom=0.16, top=0.9, wspace=0.3, hspace=0.35)
    plot_name = '../plots/calibration_' + method + ".png"

plt.savefig(plot_name, dpi=300)
plt.show()