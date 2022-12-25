import matplotlib.pyplot as plt
import tensorflow as tf
import sys
sys.path.append("/home/urz/hlichten")
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
from uncertainty.calibration_classification import reliability_diagram, uncertainty_diagram, expected_calibration_error
from functions import get_train_and_test_data, CNN

data = "cifar10"
method = "nuc"
validation = True
isontonic_reg = False

if data == "cifar10":
    fig = plt.figure(figsize=(7, 6))

    for count, model_name in enumerate(["CNN_cifar10_100", "CNN_cifar10_1000", "CNN_cifar10_10000", "CNN_cifar10"]):
        xtrain, ytrain, xval, yval, xtest, ytest, classes = get_train_and_test_data(data, validation_test_split=True)
        model = CNN(classes=classes)
        model.load_weights("../models/classification/" + model_name + "/cp.ckpt")
        ypred = model.predict(xtest)

        if model_name != "CNN_cifar10":
            num_data = int(model_name.replace('CNN_cifar10_', ""))
            xtrain = xtrain[:num_data]
            ytrain = ytrain[:num_data]

        if method == "nuc":
            if validation:
                path = "../models/classification/uncertainty_model/"
                xtrain, ytrain = xval[:int(len(xval) / 2)], yval[:int(len(yval) / 2)]
                xval, yval = xval[int(len(xval) / 2):], yval[int(len(yval) / 2):]
            else:
                path = "../models/classification/uncertainty_model/trained_on_traindata/"
            estimator = NeighborhoodUncertaintyClassifier(model, xtrain, ytrain, xval, yval, xtest,
                                                          path + model_name.replace('CNN_', "") + "/cp.ckpt")
            plt.subplot(2, 2, count+1)
            plt.title("Gesamter Trainingsdatensatz" if model_name == "CNN_cifar10" else
                                model_name.replace('CNN_cifar10_', "") + " Trainingsdaten")
            reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=ypred, certainties=estimator.certainties,
                                label_perfectly_calibrated=model_name == "CNN_cifar10_100", num_bins=15,
                                method="Prädiktionen Testdatensatz" if model_name == "CNN_cifar10_100" else None)
            ece = expected_calibration_error(tf.argmax(ytest, axis=-1), ypred, estimator.certainties).numpy()
            plt.text(0.65, 0.04, "ECE: {:.3f}".format(ece), color="brown", weight="bold")

        elif method == "mc_drop":
            print("hallo")

    if method == "nuc" and validation:
        data = "nuc_on_validation"
    plt.subplots_adjust(left=0.09, right=0.99, bottom=0.08, top=0.95, wspace=0.3, hspace=0.35)
    plt.savefig('plots/calibration_' + method + '_'+data+'.png', dpi=300)
    plt.show()
