from functions import CNN, get_train_and_test_data
from uncertainty.Ensemble import BaggingEns, DataAugmentationEns, RandomInitShuffleEns
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
from uncertainty.calibration_classification import expected_calibration_error, get_normalized_certainties
import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions

method = "nuc"

for model_name in ["CNN_cifar10_100", "CNN_cifar10_1000", "CNN_cifar10_10000", "CNN_cifar10", "CNN_cifar100"]:
    xtrain, ytrain, xval, yval, xtest, ytest, cl = get_train_and_test_data("cifar10" if model_name != "CNN_cifar100"
                                                                           else "cifar100",
                                                                           validation_test_split=True)

    model = CNN(classes=cl)
    model.load_weights("../models/classification/" + model_name + "/cp.ckpt")
    ypred = tf.argmax(model.predict(xtest), axis=-1)

    path = "../models/classification/ensembles/" + method + "/" + model_name
    if method == "mc_drop":
        estimator = MCDropoutEstimator(model, xtest, cl, xval=xval, yval=yval)
        ypred = estimator.get_ensemble_prediction()
    elif method == "softmax":
        soft_ent_uncert_val = tfd.Categorical(probs=model.predict(xval, verbose=0)).entropy().numpy()
        soft_ent_uncert_test = tfd.Categorical(probs=model.predict(xtest, verbose=0)).entropy().numpy()
        softmax_entropy = get_normalized_certainties(model.predict(xval, verbose=0), yval,
                                                     soft_ent_uncert_val, soft_ent_uncert_test)
        ece = expected_calibration_error(tf.argmax(ytest, axis=-1), ypred, softmax_entropy).numpy()
        print(model_name + " ", ece)
        continue
    elif method == "bagging":
        estimator = BaggingEns(xtest, cl, path, X_val=xval, y_val=yval, val=True)
        ypred = estimator.get_ensemble_prediction()
    elif method == "data_augmentation":
        estimator = DataAugmentationEns(xtest, cl, path, X_val=xval, y_val=yval, val=True)
        ypred = estimator.get_ensemble_prediction()
    elif method == "rand_initialization_shuffle":
        estimator = RandomInitShuffleEns(xtest, cl, path, X_val=xval, y_val=yval, val=True)
        ypred = estimator.get_ensemble_prediction()
    elif method == "nuc":
        path = "../models/classification/uncertainty_model/" + model_name + "/cp.ckpt"
        xtrain, ytrain = xval[:int(4*len(xval) / 5)], yval[:int(4*len(yval) / 5)]
        xval, yval = xval[int(4*len(xval) / 5):], yval[int(4*len(yval) / 5):]
        estimator = NeighborhoodUncertaintyClassifier(model, xtrain, ytrain, xval, yval, xtest, path)
        ece = expected_calibration_error(tf.argmax(ytest, axis=-1), ypred, estimator.certainties).numpy()

        correct = (tf.argmax(ypred, axis=-1) == tf.argmax(ytest, axis=-1))
        ece2 = tfp.stats.expected_calibration_error_quantiles(correct, estimator.certainties, num_buckets=15)
        print(model_name + " ", ece)
        print(ece2)
        continue
    else:
        raise NotImplementedError

    ece_se = expected_calibration_error(tf.argmax(ytest, axis=-1), ypred,
                                        estimator.normalized_certainties_shannon_entropy()).numpy()
    ece_mi = expected_calibration_error(tf.argmax(ytest, axis=-1), ypred,
                                        estimator.normalized_certainties_mutual_information()).numpy()
    print(model_name + " SE: ", ece_se, " MI: ", ece_mi)