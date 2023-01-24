import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import tensorflow_probability as tfp
tfd = tfp.distributions
sys.path.append("/home/urz/hlichten")
from functions import build_effnet, CNN, split_validation_from_train, get_train_and_test_data
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.Ensemble import BaggingEns, DataAugmentationEns, RandomInitShuffleEns, ENSEMBLE_LOCATION
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
from uncertainty.calibration_classification import reliability_diagram, expected_calibration_error, get_normalized_certainties



fig = plt.figure(figsize=(9, 5.6))

xtrain, ytrain, xval, yval, xtest, ytest, cl = get_train_and_test_data("cars196",
                                                                       validation_test_split=True)
model = build_effnet(cl)
model.load_weights("../models/classification/effnetb3/cp.ckpt")

# test with CNN CNN_cifar10
'''xtrain, ytrain, xval, yval, xtest, ytest, cl = get_train_and_test_data("cifar10",
                                                                       validation_test_split=True)
model = CNN(classes=cl)
model.load_weights("../models/classification/CNN_cifar10/cp.ckpt")'''

_, acc = model.evaluate(xtest, ytest, verbose=2)
print("Test accuracy: {:5.2f}%".format(100 * acc))

model_name = "effnetb3"
#model_name = "CNN_cifar10"
sampling_estimators = [MCDropoutEstimator(model, xtest, cl, xval=xval, yval=yval),
                       BaggingEns(xtest, cl, ENSEMBLE_LOCATION + "/bagging/" + model_name,
                                  X_val=xval, y_val=yval, val=True),
                       DataAugmentationEns(xtest, cl, ENSEMBLE_LOCATION + "/data_augmentation/" + model_name,
                                           X_val=xval, y_val=yval, val=True),
                       RandomInitShuffleEns(xtest, cl, ENSEMBLE_LOCATION + "/rand_initialization_shuffle/" + model_name,
                                            X_val=xval, y_val=yval, val=True)
                       ]


xtrain_nuc, ytrain_nuc, xval_nuc, yval_nuc = split_validation_from_train(xval, yval, cl, num_imgs_per_class=2)
nuc = NeighborhoodUncertaintyClassifier(model, xtrain_nuc, ytrain_nuc, xval_nuc, yval_nuc, xtest,
                                        "../models/classification/uncertainty_model/effnetb3/cp.ckpt").certainties

'''xtrain_nuc, ytrain_nuc = xval[:int(4*len(xval) / 5)], yval[:int(4*len(yval) / 5)]
xval_nuc, yval_nuc = xval[int(4*len(xval) / 5):], yval[int(4*len(yval) / 5):]
nuc = NeighborhoodUncertaintyClassifier(model, xtrain_nuc, ytrain_nuc, xval_nuc, yval_nuc, xtest,
                                        "../models/classification/uncertainty_model/CNN_cifar10/cp.ckpt").certainties'''

soft_ent_uncert_test = tfd.Categorical(probs=model.predict(xtest, verbose=0)).entropy().numpy()
soft_ent_uncert_val = tfd.Categorical(probs=model.predict(xval, verbose=0)).entropy().numpy()
softmax_entropy = get_normalized_certainties(model.predict(xval, verbose=0), yval,
                                             soft_ent_uncert_val, soft_ent_uncert_test)
model_pred = model.predict(xtest)

titles = ["MC Dropout", "Bagging", "Data Augmentation", "ZIS", "NUC", "Softmaxausgabe"]

for count, title in enumerate(titles):
    plt.subplot(2, 3, count + 1)
    plt.title(title)

    if count < 4:
        out = sampling_estimators[count].p_ens
        certs_se = sampling_estimators[count].normalized_certainties_shannon_entropy()
        certs_mi = sampling_estimators[count].normalized_certainties_mutual_information()

        reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=out, certainties=certs_se,
                            label_perfectly_calibrated=False, num_bins=10, method="SE")
        reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=out, num_bins=10, certainties=certs_mi,
                            label_perfectly_calibrated=False, color="green", method="MI")

        ece_se = expected_calibration_error(tf.argmax(ytest, axis=-1), out, certs_se).numpy()
        ece_mi = expected_calibration_error(tf.argmax(ytest, axis=-1), out, certs_mi).numpy()
        plt.text(0.02, 0.95, "ECE SE: {:.3f}".format(ece_se), color="brown", weight="bold")
        plt.text(0.02, 0.87, "ECE MI: {:.3f}".format(ece_mi), color="brown", weight="bold")

    elif count == 4:
        reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=model_pred, certainties=nuc,
                            label_perfectly_calibrated=False, num_bins=10, color="tomato", method="Certainty Score")

        ece = expected_calibration_error(tf.argmax(ytest, axis=-1), model_pred, nuc).numpy()
        plt.text(0.02, 0.95, "ECE: {:.3f}".format(ece), color="brown", weight="bold")

    else:
        reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=model_pred, certainties=softmax_entropy,
                            label_perfectly_calibrated=False, color="chocolate", num_bins=10, method="SE Softmax")

        ece = expected_calibration_error(tf.argmax(ytest, axis=-1), model_pred, softmax_entropy).numpy()
        plt.text(0.02, 0.95, "ECE: {:.3f}".format(ece), color="brown", weight="bold")

    plt.legend(loc="lower right")

plt.subplots_adjust(left=0.06, right=0.9, bottom=0.05, top=0.95, wspace=0.3, hspace=0.35)
plot_name = "../plots/calibration_effnetb3.png"
plt.savefig(plot_name, dpi=300)
plt.show()