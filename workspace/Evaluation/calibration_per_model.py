import re

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from functions import build_effnet, CNN, split_validation_from_train, get_data, COLORS, CNN_transfer_learning
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.Ensemble import BaggingEns, DataAugmentationEns, RandomInitShuffleEns, ENSEMBLE_LOCATION
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
from uncertainty.calibration_classification import reliability_diagram, expected_calibration_error, get_normalized_certainties

model_name = "CNN_cifar10_1000"
data = "cifar10"

fig = plt.figure(figsize=(9, 5.6))

num_data = int(model_name.replace('CNN_cifar10_', "")) if re.match('CNN_cifar10_.*', model_name) else None
xtrain, ytrain, xval, yval, xtest, ytest, cl = get_data(data, num_data)
num_used_classes = int(model_name.replace('CNN_cifar', "") if data != "cars196" else 100)


if model_name == "effnetb3":
    build_model_function = build_effnet
elif model_name == "CNN_cifar100":
    build_model_function = CNN
else:
    build_model_function = CNN_transfer_learning

model = tf.keras.models.load_model("../models/classification/" + model_name)

_, acc = model.evaluate(xtest, ytest, verbose=2)
print("Test accuracy: {:5.2f}%".format(100 * acc))

sampling_estimators = [MCDropoutEstimator(model, xtest, cl, xval=xval, yval=yval),
                       BaggingEns(xtest, cl, ENSEMBLE_LOCATION + "/bagging/" + model_name,
                                  X_train=xtrain, y_train=ytrain, X_val=xval, y_val=yval, val=True,
                                  build_model_function=build_model_function),
                       DataAugmentationEns(xtest, cl, ENSEMBLE_LOCATION + "/data_augmentation/" + model_name,
                                           X_train=xtrain, y_train=ytrain, X_val=xval, y_val=yval, val=True,
                                           build_model_function=build_model_function)
                       ]

if model_name != "effnetb3":
    xtrain_nuc, ytrain_nuc = xval[:int(4 * len(xval) / 5)], yval[:int(4 * len(yval) / 5)]
    xval_nuc, yval_nuc = xval[int(4 * len(xval) / 5):], yval[int(4 * len(yval) / 5):]
else:
    xtrain_nuc, ytrain_nuc, xval_nuc, yval_nuc = split_validation_from_train(xval, yval, cl, num_imgs_per_class=2)

if model_name != "CNN_cifar10_100":
    nuc = NeighborhoodUncertaintyClassifier(model, xtrain_nuc, ytrain_nuc, xval_nuc, yval_nuc, xtest,
                                        "../models/classification/certainty_model/val/10/" + model_name + "/cp.ckpt",
                                            k=10)

nuc_train = NeighborhoodUncertaintyClassifier(model, xtrain, ytrain, xval, yval, xtest,
                                        "../models/classification/certainty_model/train/10/" + model_name + "/cp.ckpt",
                                              k=10)

soft_ent_uncert_test = tfd.Categorical(probs=model.predict(xtest, verbose=0)).entropy().numpy()
soft_ent_uncert_val = tfd.Categorical(probs=model.predict(xval, verbose=0)).entropy().numpy()
softmax_entropy = get_normalized_certainties(model.predict(xval, verbose=0), yval,
                                             soft_ent_uncert_val, soft_ent_uncert_test)
model_pred = model.predict(xtest)
ax = plt.subplot(2, 3, 6)
ax.remove()

ab = ["MCD", "Bag", "DA", "NUC", "SE"]
titles = ["MC Dropout", "Bagging Ensemble", "Data Augmentation Ensemble", "NUC Methode", "Softmaxentropie"]

for count, (title, m) in enumerate(zip(titles, ab)):
    ax = plt.subplot(2, 3, count + 1)
    ax.set_axisbelow(True)
    plt.grid(visible=True, color="gainsboro", linestyle='dashed', zorder=0)
    plt.title(title)

    if count < 3:
        out = sampling_estimators[count].p_ens
        certs_se = sampling_estimators[count].normalized_certainties_shannon_entropy()
        certs_mi = sampling_estimators[count].normalized_certainties_mutual_information()

        reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=out, certainties=certs_se,
                            label_perfectly_calibrated=False, num_bins=10, method="PE", color=COLORS[m + " PE"])
        reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=out, num_bins=10, certainties=certs_mi,
                            label_perfectly_calibrated=False, color=COLORS[m + " MI"], method="MI")

        ece_se = expected_calibration_error(tf.argmax(ytest, axis=-1), tf.argmax(out, axis=-1), certs_se).numpy()
        ece_mi = expected_calibration_error(tf.argmax(ytest, axis=-1), tf.argmax(out, axis=-1), certs_mi).numpy()
        plt.text(0.02, 0.95, "ECE PE: {:.3f}".format(ece_se), color="brown", weight="bold")
        plt.text(0.02, 0.87, "ECE MI: {:.3f}".format(ece_mi), color="brown", weight="bold")

    elif count == 3:
        if model_name != "CNN_cifar10_100":
            reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=model_pred, certainties=nuc.certainties,
                                label_perfectly_calibrated=False, num_bins=10, color=COLORS[m + " Va"], method="NUC Va")
            ece_va = expected_calibration_error(tf.argmax(ytest, axis=-1), tf.argmax(model_pred, axis=-1),
                                                nuc.certainties).numpy()
            plt.text(0.02, 0.87, "ECE Va: {:.3f}".format(ece_va), color="brown", weight="bold")

        reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=model_pred, certainties=nuc_train.certainties,
                            label_perfectly_calibrated=False, num_bins=10, color=COLORS[m + " Tr"], method="NUC Tr")

        ece_tr = expected_calibration_error(tf.argmax(ytest, axis=-1), tf.argmax(model_pred, axis=-1),
                                            nuc_train.certainties).numpy()
        plt.text(0.02, 0.95, "ECE Tr: {:.3f}".format(ece_tr), color="brown", weight="bold")

    else:
        reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=model_pred, certainties=softmax_entropy,
                            label_perfectly_calibrated=False, color=COLORS[m], num_bins=10, method="SE")

        ece = expected_calibration_error(tf.argmax(ytest, axis=-1), tf.argmax(model_pred, axis=-1), softmax_entropy).numpy()
        plt.text(0.02, 0.95, "ECE: {:.3f}".format(ece), color="brown", weight="bold")

    plt.legend(loc="lower right")


plt.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.95, wspace=0.3, hspace=0.35)
ax = plt.subplot(2, 3, 5)
ax.set_position([0.56, 0.08, 0.25, 0.365])
ax = plt.subplot(2, 3, 4)
ax.set_position([0.23, 0.08, 0.25, 0.365])
plt.savefig("../plots/calibration_" + model_name + ".pdf")
plt.show()