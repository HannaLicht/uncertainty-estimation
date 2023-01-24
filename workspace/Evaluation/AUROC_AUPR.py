import json
import re
import tensorflow as tf
import sys
sys.path.append("/home/urz/hlichten")
from functions import CNN, get_train_and_test_data, build_effnet
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.Ensemble import ENSEMBLE_LOCATION, BaggingEns, DataAugmentationEns, RandomInitShuffleEns
import tensorflow_probability as tfp
tfd = tfp.distributions

"""
Calculates AUROCs and AUPRs for MC Dropout, Ensemble Methods and Softmax Shannon Entropy 
See Neighborhood_uncertainty_test.py for AUROCs and AUPRs of the NUC method
"""


SAVE_OR_USE_SAVED_MODELS = False
DATA = "cifar10"
MODEL_NAME = "CNN_cifar10_1000"
NUM_MEMBERS = 5
RUNS = 4

path_to_bagging_ens = ENSEMBLE_LOCATION + "/bagging/" + MODEL_NAME if SAVE_OR_USE_SAVED_MODELS else ""
path_to_dataAug_ens = ENSEMBLE_LOCATION + "/data_augmentation/" + MODEL_NAME if SAVE_OR_USE_SAVED_MODELS else ""
path_to_randInitShuffle_ens = ENSEMBLE_LOCATION + "/rand_initialization_shuffle/" + MODEL_NAME if \
    SAVE_OR_USE_SAVED_MODELS else ""
model_path = "../models/classification/" + MODEL_NAME + "/cp.ckpt"


def auroc(lbls_test, preds_test, uncerts):
    m = tf.keras.metrics.AUC(curve='ROC')
    m.update_state((lbls_test != preds_test), uncerts)      # incorrect: label 0 -> uncertain = positive class
    return m.result().numpy()


def aupr(lbls_test, preds_test, uncerts):
    m = tf.keras.metrics.AUC(curve="PR")
    m.update_state((lbls_test != preds_test), uncerts)      # incorrect: label 0 -> uncertain = positive class
    return m.result().numpy()


x, y, x_val, y_val, x_test, y_test, num_classes = get_train_and_test_data(DATA, validation_test_split=True)

model = build_effnet(num_classes) if MODEL_NAME == "effnetb3" else CNN(classes=num_classes)
model.load_weights(model_path)

num_data = None
if re.match('CNN_cifar10_.*', MODEL_NAME):
    num_data = int(MODEL_NAME.replace('CNN_cifar10_', ""))
    x, y = x[:num_data], y[:num_data]

lbls = tf.math.argmax(y_test, axis=-1).numpy()
y_pred = tf.math.argmax(model.predict(x_test), axis=-1).numpy()

_, acc = model.evaluate(x, y)
print("Accuracy on train dataset: ", acc)
_, acc = model.evaluate(x_test, y_test)
print("Accuracy on test dataset: ", acc)

for _ in range(RUNS):

    model = build_effnet(num_classes) if MODEL_NAME == "effnetb3" else CNN(classes=num_classes)
    model.load_weights(model_path)

    MCEstimator = MCDropoutEstimator(model, x_test, num_classes, T=50)
    DAEstimator = DataAugmentationEns(x_test, num_classes, model_name=MODEL_NAME, X_train=x, y_train=y,
                                      path_to_ensemble=path_to_dataAug_ens, num_members=NUM_MEMBERS,
                                      X_val=x_val, y_val=y_val)
    RISEstimator = RandomInitShuffleEns(x_test, num_classes, model_name=MODEL_NAME,  X_train=x, y_train=y,
                                        path_to_ensemble=path_to_randInitShuffle_ens, num_members=NUM_MEMBERS,
                                        X_val=x_val, y_val=y_val)
    BaEstimator = BaggingEns(x_test, num_classes, model_name=MODEL_NAME, path_to_ensemble=path_to_bagging_ens,
                             X_train=x, y_train=y, num_members=NUM_MEMBERS, X_val=x_val, y_val=y_val)

    methods = ["Softmax",
               "MCdrop SE", "MCdrop MI",
               "Bag SE", "Bag MI", "Rand SE", "Rand MI",
               "DataAug SE", "DataAug MI"
               ]

    y_pred_drop = MCEstimator.get_ensemble_prediction()
    y_pred_bag = BaEstimator.get_ensemble_prediction()
    y_pred_aug = DAEstimator.get_ensemble_prediction()
    y_pred_rand = RISEstimator.get_ensemble_prediction()

    preds = [y_pred,
             y_pred_drop, y_pred_drop,
             y_pred_bag, y_pred_bag,
             y_pred_rand, y_pred_rand,
             y_pred_aug, y_pred_aug
             ]

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
    uncertainties = [soft_ent_uncert_test/tf.reduce_max(soft_ent_uncert_test),
                     mcdr_se/tf.reduce_max(mcdr_se), mcdr_mi/tf.reduce_max(mcdr_mi),
                     bag_se/tf.reduce_max(bag_se), bag_mi/tf.reduce_max(bag_mi),
                     rand_se/tf.reduce_max(rand_se), rand_mi/tf.reduce_max(rand_mi),
                     aug_se/tf.reduce_max(aug_se), aug_mi/tf.reduce_max(aug_mi)]

    for i, (uncert, pred) in enumerate(zip(uncertainties, preds)):
        roc = auroc(lbls, pred, uncert)
        pr = aupr(lbls, pred, uncert)
        print(methods[i])
        print("AUROC: ", roc)
        print("AUPR: ", pr, "\n")

        with open('results_auroc_aupr.json') as json_file:
            data = json.load(json_file)
            data[methods[i]][MODEL_NAME]["auroc"] = data[methods[i]][MODEL_NAME]["auroc"] + [roc.item()]
            data[methods[i]][MODEL_NAME]["aupr"] = data[methods[i]][MODEL_NAME]["aupr"] + [pr.item()]
        with open('results_auroc_aupr.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)


with open('results_auroc_aupr.json') as json_file:
    data = json.load(json_file)

# print table of AUROCs and AUPRs
print("AUROCs and AUPRs for model " + MODEL_NAME)
print("Method\tAUROC\tAUPR")
for method in data.keys():
    a_roc = tf.reduce_mean(data[method][MODEL_NAME]["auroc"])
    a_pr = tf.reduce_mean(data[method][MODEL_NAME]["aupr"])
    a_roc, a_pr = (a_roc[1], a_pr[1]) if method == "nuc_train" or method == "nuc_val" else (a_roc, a_pr)
    print(method, "\t", a_roc, "\t", a_pr)