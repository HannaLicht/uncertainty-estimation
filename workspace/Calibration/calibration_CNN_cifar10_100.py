import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

sys.path.append("/home/urz/hlichten")
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.Ensemble import DataAugmentationEns, RandomInitShuffleEns, ENSEMBLE_LOCATION
from uncertainty.calibration_classification import  plot_regression, uncertainty_diagram
from functions import get_train_and_test_data, CNN


fig = plt.figure(figsize=(9, 2.8))
xtrain, ytrain, xval, yval, xtest, ytest, cl = get_train_and_test_data("cifar10", validation_test_split=True)
model = CNN(classes=cl)
model.load_weights("../models/classification/CNN_cifar10_100/cp.ckpt")

for count, (method, title) in enumerate(zip(["data_augmentation", "rand_initialization_shuffle", "mc_drop"],
                                        ["Data Augmentaion", "ZIS", "MC Dropout"])):

    plt.subplot(1, 3, count + 1)
    path = ENSEMBLE_LOCATION + "/" + method + "/CNN_cifar10_100"

    if method == "mc_drop":
        estimator = MCDropoutEstimator(model, xtest, cl, xval=xval, yval=yval)
    elif method == "data_augmentation":
        estimator = DataAugmentationEns(xtest, cl, path, X_val=xval, y_val=yval, val=True)
    elif method == "rand_initialization_shuffle":
        estimator = RandomInitShuffleEns(xtest, cl, path, X_val=xval, y_val=yval, val=True)
    else:
        raise NotImplementedError

    plt.title(title)
    regressor = plot_regression(tf.argmax(yval, axis=-1), estimator.val_p_ens,
                                estimator.uncertainties_mutual_information(val=True),
                                label=True if title == "MC Dropout" else False)
    uncertainty_diagram(tf.argmax(ytest, axis=-1), estimator.p_ens,
                        estimator.uncertainties_mutual_information(), title=title,
                        label="Testdaten" if title == "MC Dropout" else None, color="green")
    plt.xlabel("Mutual Information")

    if method == "mc_drop":
        start = 0
        stop = 0.5
        step = 0.15
    else:
        start = 0
        stop = 0.65
        step = 0.2

plt.xticks(np.arange(start, stop, step=step))

plt.subplots_adjust(left=0.06, right=0.89, bottom=0.16, top=0.9, wspace=0.3, hspace=0.35)
plot_name = "../plots/calibration_CNN_c10_100.png"

plt.savefig(plot_name, dpi=300)
plt.show()