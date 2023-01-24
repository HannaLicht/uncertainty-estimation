from matplotlib import pyplot as plt
import re
import keras.applications.efficientnet as efn
from functions import create_simple_model, get_test_data, CNN, get_train_and_test_data
from uncertainty.Ensemble import ENSEMBLE_LOCATION, BaggingEns
from uncertainty.MC_Dropout import MCDropoutEstimator
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

M = 5
T = 50
METHOD = "bagging"


def uncertainty_front(uncertainties1, uncertainties2, name1: str, name2: str):
    plt.scatter(uncertainties1, uncertainties2, s=8)
    plt.xlabel(name1)
    plt.ylabel(name2)


plt.figure(figsize=(18, 20))
for count, model_name in enumerate(["CNN_cifar10_1000", "CNN_cifar10_10000", "CNN_cifar10"]):
    checkpoint_path = "../models/classification/" + model_name + "/cp.ckpt"
    model = CNN(classes=10)
    model.load_weights(checkpoint_path)

    if METHOD == "MC drop":
        x, y, num_classes = get_test_data("cifar10")
        estimator = MCDropoutEstimator(model, x, num_classes, T)
    elif METHOD == "bagging":
        path_to_ensemble = ENSEMBLE_LOCATION + "/bagging/" + model_name
        X_train, y_train, x, y, classes = get_train_and_test_data("cifar10")
        estimator = BaggingEns(x, classes, model_name=model_name,
                               path_to_ensemble=path_to_ensemble, num_members=M)
    else:
        raise NotImplementedError

    method_se = estimator.uncertainties_shannon_entropy()
    method_mi = estimator.uncertainties_mutual_information()
    singlepred_se = tfd.Categorical(probs=model.predict(x, verbose=0)).entropy().numpy()

    plt.subplot(4, 3, 3*count + 1)
    uncertainty_front(singlepred_se, method_se, "shannon entropy single prediction", "shannon entropy " + METHOD)
    plt.subplot(4, 3, 3*count + 2)
    uncertainty_front(method_se, method_mi, "shannon entropy " + METHOD, "mutual information " + METHOD)
    plt.subplot(4, 3, 3*count + 3)
    uncertainty_front(singlepred_se, method_mi, "shannon entropy single prediction", "mutual information " + METHOD)
    plt.suptitle(model_name)

    plt.subplot(4, 3, 10 + count)
    plt.boxplot([singlepred_se, method_se, method_mi], showfliers=False)
    plt.xticks([1.0, 2.0, 3.0], ["single SE", "MC drop SE", "MC drop MI"])
    plt.xlabel("Method")
    plt.ylabel("Uncertainty")
    plt.title(model_name)

plt.show()
