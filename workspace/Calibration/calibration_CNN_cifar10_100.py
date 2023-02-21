import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

sys.path.append("/home/urz/hlichten")
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.Ensemble import DataAugmentationEns, ENSEMBLE_LOCATION, BaggingEns
from uncertainty.calibration_classification import plot_regression, uncertainty_diagram, reliability_diagram, \
    expected_calibration_error
from functions import get_data, CNN, COLORS, CNN_transfer_learning

fig = plt.figure(figsize=(8.8, 5.6))
xtrain, ytrain, xval, yval, xtest, ytest, cl = get_data("cifar10", num_data=100)
model = tf.keras.models.load_model("../models/classification/CNN_cifar10_100")

path_da = ENSEMBLE_LOCATION + "/data_augmentation/CNN_cifar10_100"
path_bag = ENSEMBLE_LOCATION + "/bagging/CNN_cifar10_100"

est = [MCDropoutEstimator(model, xtest, cl, xval=xval, yval=yval),
       BaggingEns(xtest, cl, path_bag, X_train=xtrain, y_train=ytrain, X_val=xval, y_val=yval, val=True, build_model_function=CNN_transfer_learning),
       DataAugmentationEns(xtest, cl, path_da, X_train=xtrain, y_train=ytrain, X_val=xval, y_val=yval, val=True, build_model_function=CNN_transfer_learning)]
colors = [COLORS["MCD MI"], COLORS["Bag MI"], COLORS["DA MI"]]

for count, (estimator, title, c) in enumerate(zip(est, ["MC Dropout", "Bagging", "Data Aug."], colors)):

    ax = plt.subplot(2, 3, count + 1)
    ax.set_axisbelow(True)
    plt.grid(visible=True, color="gainsboro", linestyle='dashed', zorder=0)
    plot_regression(tf.argmax(yval, axis=-1), estimator.val_p_ens,
                    estimator.uncertainties_mutual_information(val=True), label=True,
                    utest=estimator.uncertainties_mutual_information())
    uncertainty_diagram(tf.argmax(ytest, axis=-1), estimator.p_ens,
                        estimator.uncertainties_mutual_information(), title="G " + title, color=c, label="Testprädiktionen")
    plt.xlabel("Mutual Information")
    plt.legend(loc="upper right")

    ax = plt.subplot(2, 3, count + 4)
    ax.set_axisbelow(True)
    plt.grid(visible=True, color="gainsboro", linestyle='dashed', zorder=0)
    plt.title("Kalibrierung " + title)
    reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=estimator.p_ens,
                        certainties=estimator.normalized_certainties_mutual_information(),
                        label_perfectly_calibrated=False, color=c, num_bins=10)
    ece = expected_calibration_error(tf.argmax(ytest, axis=-1), tf.argmax(model.predict(xtest), axis=-1),
                                     estimator.normalized_certainties_mutual_information()).numpy()
    plt.text(0.02, 0.95, "ECE: {:.3f}".format(ece), color="brown", weight="bold")
    plt.xlabel("Mutual Information")

plt.subplots_adjust(left=0.06, right=0.98, bottom=0.1, top=0.94, wspace=0.3, hspace=0.4)
plot_name = "../plots/mutual_information_CNN_c10_100.png"

plt.savefig(plot_name, dpi=300)
plt.show()