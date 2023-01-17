import matplotlib.pyplot as plt
import tensorflow as tf
from uncertainty.calibration_classification import get_normalized_certainties
import sys
sys.path.append("/home/urz/hlichten")
sys.path.append("/home/hanna/Schreibtisch/Ingenieurinformatik VW/Igenieurinformatik/BA/uncertainty-estimation/workspace")
print(sys.path)

from functions import CNN, get_train_and_test_data, split_validation_from_train
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.Ensemble import ENSEMBLE_LOCATION, BaggingEns, DataAugmentationEns, RandomInitShuffleEns
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
import tensorflow_probability as tfp
tfd = tfp.distributions

DATA = "cifar10_100"
MODEL = "CNN"
CHECKPOINT_PATH = "../models/classification/" + MODEL + "_" + DATA + "/cp.ckpt"
NUM_MEMBERS = 5

path_to_bagging_ens = ENSEMBLE_LOCATION + "/bagging/" + MODEL + "_" + DATA
path_to_dataAug_ens = ENSEMBLE_LOCATION + "/data_augmentation/" + MODEL + "_" + DATA
path_to_randInitShuffle_ens = ENSEMBLE_LOCATION + "/rand_initialization_shuffle/" + MODEL + "_" + DATA
path_uncertainty_model = "../models/classification/uncertainty_model/" + DATA + "/cp.ckpt"
path_uncertainty_model_on_train = "../models/classification/uncertainty_model/trained_on_traindata/" + DATA + "/cp.ckpt"


def auroc(lbls_test, preds_test, certs):
    m = tf.keras.metrics.AUC(curve='ROC')
    m.update_state((lbls_test == preds_test), certs)
    return m.result().numpy()


def aupr(lbls_test, preds_test, certs):
    m = tf.keras.metrics.AUC(curve="PR")
    m.update_state((lbls_test == preds_test), certs)
    return m.result().numpy()


model = CNN(classes=100 if DATA == "cifar100" else 10)
model.load_weights(CHECKPOINT_PATH)

x, y, x_val, y_val, x_test, y_test, num_classes = get_train_and_test_data("cifar10" if DATA == "cifar10_1000" or
                                                                                       DATA == "cifar10_10000" or
                                                                                       DATA == "cifar10_100" else DATA,
                                                                          validation_test_split=True)
if DATA == "cifar10_1000":
    x, y = x[:1000], y[:1000]
elif DATA == "cifar10_10000":
    x, y = x[:10000], y[:10000]
elif DATA == "cifar10_100":
    x, y = x[:100], y[:100]

lbls = tf.math.argmax(y_test, axis=-1).numpy()
y_pred = tf.math.argmax(model.predict(x_test), axis=-1).numpy()

_, acc = model.evaluate(x, y)
print("Accuracy on train dataset: ", acc)
_, acc = model.evaluate(x_test, y_test)
print("Accuracy on test dataset: ", acc)

model_name = MODEL + "_cifar10" if DATA == "cifar10_1000" or DATA == "cifar10_10000" or DATA == "cifar10_100" \
    else MODEL + "_" + DATA


MCEstimator = MCDropoutEstimator(model, x_test, num_classes, T=50)
DAEstimator = DataAugmentationEns(x, y, x_test, num_classes, model_name=model_name,
                                  path_to_ensemble=path_to_dataAug_ens, num_members=NUM_MEMBERS,
                                  X_val=x_val, y_val=y_val)
RISEstimator = RandomInitShuffleEns(x, y, x_test, num_classes, model_name=model_name,
                                    path_to_ensemble=path_to_randInitShuffle_ens, num_members=NUM_MEMBERS,
                                    X_val=x_val, y_val=y_val)
BaEstimator = BaggingEns(x, y, x_test, num_classes, model_name=model_name, path_to_ensemble=path_to_bagging_ens,
                         num_members=NUM_MEMBERS, X_val=x_val, y_val=y_val)

if MODEL == "effnetb3":
    nuc_xtrain, nuc_ytrain, nuc_xval, nuc_yval = split_validation_from_train(x_val, y_val, num_classes,
                                                                             num_imgs_per_class=2)
else:
    nuc_xtrain, nuc_ytrain = x_val[:int(4*len(x_val) / 5)], y_val[:int(4*len(y_val) / 5)]
    nuc_xval, nuc_yval = x_val[int(4*len(x_val) / 5):], y_val[int(4*len(y_val) / 5):]

NUEstimator = NeighborhoodUncertaintyClassifier(model, nuc_xtrain, nuc_ytrain, nuc_xval, nuc_yval, x_test,
                                                path_uncertainty_model=path_uncertainty_model)
NUEstimator_on_train = NeighborhoodUncertaintyClassifier(model, x, y, x_val, y_val, x_test,
                                                         path_uncertainty_model=path_uncertainty_model_on_train)

methods = ["MCdrop SE", "MCdrop MI",
           "Bag SE", "Bag MI", "Rand SE", "Rand MI",
           "DataAug SE", "DataAug MI",
           "NUC_train", "NUC_valid", "Softmax"]

y_pred_drop = MCEstimator.get_ensemble_prediction()
y_pred_bag = BaEstimator.get_ensemble_prediction()
y_pred_aug = DAEstimator.get_ensemble_prediction()
y_pred_rand = RISEstimator.get_ensemble_prediction()
preds = [y_pred_drop, y_pred_drop, y_pred_bag, y_pred_bag, y_pred_rand, y_pred_rand,
         y_pred_aug, y_pred_aug,
         y_pred, y_pred, y_pred]

soft_ent_uncert_test = tfd.Categorical(probs=model.predict(x_test, verbose=0)).entropy().numpy()

mcdr_se = MCEstimator.uncertainties_shannon_entropy()
mcdr_mi = MCEstimator.uncertainties_mutual_information()
bag_se = BaEstimator.uncertainties_shannon_entropy()
bag_mi = BaEstimator.uncertainties_mutual_information()
rand_se = RISEstimator.uncertainties_shannon_entropy()
rand_mi = RISEstimator.uncertainties_mutual_information()
aug_se = DAEstimator.uncertainties_shannon_entropy()
aug_mi = DAEstimator.uncertainties_mutual_information()

# make certainties between 0 and 1
certainties = [1 - mcdr_se/tf.reduce_max(mcdr_se), 1 - mcdr_mi/tf.reduce_max(mcdr_mi),
               1 - bag_se/tf.reduce_max(bag_se), 1 - bag_mi/tf.reduce_max(bag_mi),
               1 - rand_se/tf.reduce_max(rand_se), 1 - rand_mi/tf.reduce_max(rand_mi),
               1 - aug_se/tf.reduce_max(aug_se), 1 - aug_mi/tf.reduce_max(aug_mi),
               NUEstimator_on_train.certainties, NUEstimator.certainties,
               1 - soft_ent_uncert_test/tf.reduce_max(soft_ent_uncert_test)
               ]

print(methods)
print([auroc(lbls, pred, cert) for cert, pred in zip(certainties, preds)])
print([aupr(lbls, pred, cert) for cert, pred in zip(certainties, preds)])
