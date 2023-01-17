import matplotlib.pyplot as plt
import tensorflow as tf
import sys
sys.path.append("/home/urz/hlichten")
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
from uncertainty.calibration_classification import reliability_diagram, expected_calibration_error
from functions import get_train_and_test_data, CNN


validation = True

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
        xtrain, ytrain = xval[:int(4*len(xval) / 5)], yval[:int(4*len(yval) / 5)]
        xval, yval = xval[int(4*len(xval) / 5):], yval[int(4*len(yval) / 5):]
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
plot_name = '../plots/calibration_nuc_on_validation_cifar10.png' if validation else 'plots/calibration_nuc_cifar10.png'

plt.savefig(plot_name, dpi=300)
plt.show()