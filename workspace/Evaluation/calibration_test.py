import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import keras.applications.efficientnet as efn
sys.path.append("/home/urz/hlichten")
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.Ensemble import BaggingEns, DataAugmentationEns, RandomInitShuffleEns, ENSEMBLE_LOCATION
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
from uncertainty.calibration_classification import reliability_diagram, uncertainty_diagram, expected_calibration_error
from functions import get_train_and_test_data, CNN

method = "rand_initialization_shuffle"
metric = "SE"
validation = True
isontonic_reg = False
effnet = False


def calibration(model, model_name, title, xtrain, ytrain, xval, yval, xtest, ytest, cl):
    path = ENSEMBLE_LOCATION + "/" + method + "/" + model_name

    if method == "mc_drop":
        estimator = MCDropoutEstimator(model, xtest, cl, xval=xval, yval=yval)
    elif method == "bagging":
        estimator = BaggingEns(xtrain, ytrain, xtest, cl, model_name, path, xval, yval, val=True)
    elif method == "data_augmentation":
        estimator = DataAugmentationEns(xtrain, ytrain, xtest, cl, model_name, path, xval, yval, val=True)
    elif method == "rand_initialization_shuffle":
        estimator = BaggingEns(xtrain, ytrain, xtest, cl, model_name, path, xval, yval, val=True)
    else:
        raise NotImplementedError

    if isontonic_reg:
        # xline = 0.2 if metric == "MI" else 1.0
        if metric == "SE":
            uncertainty_diagram(tf.argmax(ytest, axis=-1), estimator.p_ens,
                                estimator.uncertainties_shannon_entropy(),
                                label="SE Testdaten" if model_name == "CNN_cifar100" else None)
            uncertainty_diagram(tf.argmax(yval, axis=-1), estimator.val_p_ens,
                                estimator.uncertainties_shannon_entropy(val=True), title=title,
                                label="SE Validierungsdaten" if model_name == "CNN_cifar100" else None,
                                plot_reg=True)
        else:
            uncertainty_diagram(tf.argmax(ytest, axis=-1), estimator.p_ens,
                                estimator.uncertainties_mutual_information(),
                                label="MI Testdaten" if model_name == "CNN_cifar100" else None, color="green")
            uncertainty_diagram(tf.argmax(yval, axis=-1), estimator.val_p_ens,
                                estimator.uncertainties_mutual_information(val=True), title=title,
                                label="MI Validierungsdaten" if model_name == "CNN_cifar100" else None,
                                plot_reg=True, color="salmon")

    else:
        plt.title(title)
        reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=estimator.p_ens,
                            certainties=estimator.normalized_certainties_shannon_entropy(),
                            label_perfectly_calibrated=False, num_bins=10,
                            method="Shannon Entropy" if model_name == "CNN_cifar100" else None)
        reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=estimator.p_ens, num_bins=10,
                            certainties=estimator.normalized_certainties_mutual_information(),
                            label_perfectly_calibrated=model_name == "CNN_cifar100", color="green",
                            method="Mutual Information" if model_name == "CNN_cifar100" else None)

        ece_se = expected_calibration_error(tf.argmax(ytest, axis=-1), estimator.p_ens,
                                            estimator.normalized_certainties_shannon_entropy()).numpy()
        ece_mi = expected_calibration_error(tf.argmax(ytest, axis=-1), estimator.p_ens,
                                            estimator.normalized_certainties_mutual_information()).numpy()
        plt.text(0.02, 0.95, "ECE SE: {:.3f}".format(ece_se), color="brown", weight="bold")
        plt.text(0.02, 0.87, "ECE MI: {:.3f}".format(ece_mi), color="brown", weight="bold")


if method == "nuc":
    fig = plt.figure(figsize=(9, 2.8))

    for count, model_name in enumerate(["CNN_cifar10_100", "CNN_cifar10_1000", "CNN_cifar10"]):
        xtrain, ytrain, xval, yval, xtest, ytest, classes = get_train_and_test_data("cifar10", validation_test_split=True)
        model = CNN(classes=classes)
        model.load_weights("../models/classification/" + model_name + "/cp.ckpt")
        ypred = model.predict(xtest)
        plt.subplot(1, 3, count + 1)

        if model_name != "CNN_cifar10":
            num_data = int(model_name.replace('CNN_cifar10_', ""))
            xtrain = xtrain[:num_data]
            ytrain = ytrain[:num_data]

        if validation:
            path = "../models/classification/uncertainty_model/"
            xtrain, ytrain = xval[:int(len(xval) / 2)], yval[:int(len(yval) / 2)]
            xval, yval = xval[int(len(xval) / 2):], yval[int(len(yval) / 2):]
        else:
            path = "../models/classification/uncertainty_model/trained_on_traindata/"

        estimator = NeighborhoodUncertaintyClassifier(model, xtrain, ytrain, xval, yval, xtest,
                                                      path + model_name.replace('CNN_', "") + "/cp.ckpt")
        plt.title("Gesamter Trainingsdatensatz" if model_name == "CNN_cifar10" else
                                model_name.replace('CNN_cifar10_', "") + " Trainingsdaten")
        reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=ypred, certainties=estimator.certainties,
                            label_perfectly_calibrated=model_name == "CNN_cifar10", num_bins=15,
                            method="Prädiktionen Testdaten" if model_name == "CNN_cifar10" else None)
        ece = expected_calibration_error(tf.argmax(ytest, axis=-1), ypred, estimator.certainties).numpy()
        plt.text(0.02, 0.95, "ECE: {:.3f}".format(ece), color="brown", weight="bold")

    plt.subplots_adjust(left=0.06, right=0.88, bottom=0.16, top=0.9, wspace=0.3, hspace=0.35)
    plot_name = 'plots/calibration_nuc_on_validation_cifar10.png' if validation else 'plots/calibration_nuc_cifar10.png'

elif method != "nuc":
    fig = plt.figure(figsize=(9, 2.8))

    if effnet:
        xtrain, ytrain, xval, yval, xtest, ytest, cl = get_train_and_test_data("imagenet",
                                                                               validation_test_split=True)
        eff = efn.EfficientNetB3(weights='imagenet')
        calibration(eff, "EfficientNetB3", "EfficientNetB3 ILSVRC2012", xtrain, ytrain, xval, yval, xtest, ytest, cl)

    else:
        for count, (model_name, title) in enumerate(zip(["CNN_cifar10_100", "CNN_cifar10", "CNN_cifar100"],
                                                ["100 Trainingsdaten Cifar10", "Cifar10 gesamt", "Cifar100 gesamt"])):

            xtrain, ytrain, xval, yval, xtest, ytest, cl = get_train_and_test_data("cifar10" if count != 2 else "cifar100",
                                                                                   validation_test_split=True)
            model = CNN(classes=cl)
            model.load_weights("../models/classification/" + model_name + "/cp.ckpt")
            plt.subplot(1, 3, count + 1)
            calibration(model, model_name, title, xtrain, ytrain, xval, yval, xtest, ytest, cl)

    if isontonic_reg:
        plot_name = 'plots/calibration_' + method + "_" + metric + "_isotonic_regression.png"
        plt.subplots_adjust(left=0.06, right=0.88, bottom=0.16, top=0.9, wspace=0.3, hspace=0.35)
    else:
        plt.subplots_adjust(left=0.06, right=0.9, bottom=0.16, top=0.9, wspace=0.3, hspace=0.35)
        plot_name = 'plots/calibration_' + method + ".png"

plt.savefig(plot_name, dpi=300)
plt.show()
