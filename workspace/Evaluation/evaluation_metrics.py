import json
import re
import time
import numpy as np
import tensorflow as tf
import sys
sys.path.append("/home/urz/hlichten")
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
from uncertainty.calibration_classification import get_normalized_certainties, expected_calibration_error
from functions import CNN, get_data, build_effnet, split_validation_from_train
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
MODEL_NAME = "CNN_cifar10_10000"
RUNS = 9

path_to_bagging_ens = ENSEMBLE_LOCATION + "/bagging/" + MODEL_NAME if SAVE_OR_USE_SAVED_MODELS else ""
path_to_dataAug_ens = ENSEMBLE_LOCATION + "/data_augmentation/" + MODEL_NAME if SAVE_OR_USE_SAVED_MODELS else ""
path_to_randInitShuffle_ens = ENSEMBLE_LOCATION + "/rand_initialization_shuffle/" + MODEL_NAME if \
    SAVE_OR_USE_SAVED_MODELS else ""
model_path = "../models/classification/" + MODEL_NAME + "/cp.ckpt"

k_val, k_tr = 3, 3

if MODEL_NAME == "CNN_cifar10_100":
    k_tr = 50
if MODEL_NAME == "CNN_cifar10_1000":
    k_tr = 25
    k_val = 5

pre_path_uncertainty_model = "../models/classification/uncertainty_model/"
path_uncertainty_model = pre_path_uncertainty_model + "val/" + str(k_val) + "/" + MODEL_NAME + "/cp.ckpt" \
    if SAVE_OR_USE_SAVED_MODELS else None
path_uncertainty_model_on_train = pre_path_uncertainty_model + "train/" + MODEL_NAME + "/cp.ckpt" \
    if SAVE_OR_USE_SAVED_MODELS else None


class Evaluator:

    def __init__(self, lbls_test, preds_test, certs):
        self.certainties = certs
        self.correct = (lbls_test == preds_test)

    def make_groups(self, threshold):
        """
        :param threshold: certainties above it are classified as certain and those below are classified as uncertain
        :return: True Uncertain (TU) = incorrect prediction the model is unsure about,
                 True Certain (TC) = correct prediction the model is sure about,
                 False Uncertain (FU) = correct prediction the model is uncertain about,
                 False Certain (FC) = incorrect prediction the model is certain about (worst case)
        """
        TU, TC, FU, FC = 0, 0, 0, 0
        for truth, certainty in zip(self.correct, self.certainties):
            if certainty >= threshold:
                if truth: TC += 1
                else: FC += 1
            else:
                if truth: FU += 1
                else: TU += 1
        return TU, TC, FU, FC

    def recall(self, groups):
        """
        = sensitivity
        indicates how much of the positive outputs are correctly labeled as positive:
        ratio of correct predictions that are certain (TC) to all the correct predictions (TC+FU)
        :param groups: True Uncertain (TU), True Certain (TC), False Uncertain (FU), False Certain (FC)
        :return: sensitivity ratio
        """
        TU = groups[0]
        FC = groups[3]
        return TU/(TU+FC)

    def specificity(self, groups):
        """
        true negative rate:
        ratio of incorrect predictions that are uncertain (TU) to all the incorrect predictions (TU+FC)
        :param groups: True Uncertain (TU), True Certain (TC), False Uncertain (FU), False Certain (FC)
        :return: specificity ratio
        """
        TC = groups[1]
        FU = groups[2]
        return TC/(TC + FU)

    def precision(self, groups):
        """
        positive predictive value:
        ratio of certain and correct (TC) predictions to all certain predictions (TC+FC)
        :param groups: True Uncertain (TU), True Certain (TC), False Uncertain (FU), False Certain (FC)
        :return: precision ratio
        """
        TU = groups[0]
        FU = groups[2]
        try:
            result = TU/(TU+FU)
            return result
        except ZeroDivisionError:
            return 1.

    def results(self, thresholds):
        preci, speci, rec = [], [], []
        for tr in thresholds:
            gr = self.make_groups(tr)
            preci.append(self.precision(gr))
            speci.append(self.specificity(gr))
            rec.append(self.recall(gr))
        return preci, speci, rec


def auroc(lbls_test, preds_test, uncerts):
    m = tf.keras.metrics.AUC(curve='ROC')
    m.update_state((lbls_test != preds_test), uncerts)      # incorrect: label 1 -> uncertain = positive class
    return m.result().numpy()


def aupr(lbls_test, preds_test, uncerts):
    m = tf.keras.metrics.AUC(curve="PR")
    m.update_state((lbls_test != preds_test), uncerts)      # incorrect: label 1 -> uncertain = positive class
    return m.result().numpy()


def calculate_metrics(test_lbls, pred, uncert):
    thresholds = list(np.linspace(0, 1, 200))
    pr = tf.metrics.Precision(thresholds=thresholds)
    rec = tf.metrics.Recall(thresholds=thresholds)
    fp = tf.metrics.FalsePositives(thresholds=thresholds)
    tn = tf.metrics.TrueNegatives(thresholds=thresholds)

    incorrect = (test_lbls != pred)
    pr.update_state(incorrect, uncert)
    rec.update_state(incorrect, uncert)
    fp.update_state(incorrect, uncert)
    tn.update_state(incorrect, uncert)

    spe = (tn.result() / (fp.result() + tn.result())).numpy()

    pre = [p.item() for p in pr.result().numpy()]
    spe = [s.item() for s in spe]
    rec = [r.item() for r in rec.result().numpy()]

    return pre, spe, rec


num_data = None
if re.match('CNN_cifar10_.*', MODEL_NAME):
    num_data = int(MODEL_NAME.replace('CNN_cifar10_', ""))
x, y, x_val, y_val, x_test, y_test, num_classes, _, _ = get_data(DATA, num_data=num_data)

model = build_effnet(num_classes) if MODEL_NAME == "effnetb3" else CNN(classes=num_classes)
model.load_weights(model_path)

lbls = tf.math.argmax(y_test, axis=-1).numpy()
y_pred = tf.math.argmax(model.predict(x_test), axis=-1).numpy()

_, acc = model.evaluate(x, y)
print("Accuracy on train dataset: ", acc)
_, acc = model.evaluate(x_test, y_test)
print("Accuracy on test dataset: ", acc)

for _ in range(RUNS):

    model = build_effnet(num_classes) if MODEL_NAME == "effnetb3" else CNN(classes=num_classes)
    model.load_weights(model_path)

    model_pred = model.predict(x_test, verbose=0)

    st = time.time()
    soft_ent_uncert_test = tfd.Categorical(probs=model_pred).entropy().numpy()
    end1 = time.time()

    soft_ent_uncert_val = tfd.Categorical(probs=model.predict(x_val, verbose=0)).entropy().numpy()
    softmax_entropy = get_normalized_certainties(model.predict(x_val, verbose=0), y_val,
                                                 soft_ent_uncert_val, soft_ent_uncert_test)
    end2 = time.time()
    t_soft_uncert = round(end1 - st, 5)
    t_soft_calib = round(end2 - st, 5)

    st = time.time()
    MCEstimator = MCDropoutEstimator(model, x_test, num_classes, T=50, xval=x_val, yval=y_val)
    end = time.time()
    t_drop_calib = round(end - st, 5)

    st = time.time()
    DAEstimator = DataAugmentationEns(x_test, num_classes, model_name=MODEL_NAME, X_train=x, y_train=y,
                                      path_to_ensemble=path_to_dataAug_ens, X_val=x_val, y_val=y_val, val=True)
    end = time.time()
    t_da = round(end - st, 5)

    st = time.time()
    RISEstimator = RandomInitShuffleEns(x_test, num_classes, model_name=MODEL_NAME,  X_train=x, y_train=y,
                                        path_to_ensemble=path_to_randInitShuffle_ens, X_val=x_val, y_val=y_val, val=True)
    end = time.time()
    t_ris = round(end - st, 5)

    st = time.time()
    BaEstimator = BaggingEns(x_test, num_classes, model_name=MODEL_NAME, path_to_ensemble=path_to_bagging_ens,
                             X_train=x, y_train=y, X_val=x_val, y_val=y_val, val=True)
    end = time.time()
    t_bag = round(end - st, 5)

    st = time.time()
    NUEstimator_on_train = NeighborhoodUncertaintyClassifier(model, x, y, x_val, y_val, x_test, k=k_tr,
                                                             path_uncertainty_model=path_uncertainty_model_on_train)
    end = time.time()
    t_nuc_tr = round(end - st, 5)

    if MODEL_NAME == "effnetb3":
        nuc_xtrain, nuc_ytrain, nuc_xval, nuc_yval = split_validation_from_train(x_val, y_val, num_classes,
                                                                                 num_imgs_per_class=2)
    else:
        nuc_xtrain, nuc_ytrain = x_val[:int(4 * len(x_val) / 5)], y_val[:int(4 * len(y_val) / 5)]
        nuc_xval, nuc_yval = x_val[int(4 * len(x_val) / 5):], y_val[int(4 * len(y_val) / 5):]

    if MODEL_NAME != "CNN_cifar10_100":
        st = time.time()
        NUEstimator = NeighborhoodUncertaintyClassifier(model, nuc_xtrain, nuc_ytrain, nuc_xval, nuc_yval, x_test,
                                                        path_uncertainty_model=path_uncertainty_model, k=k_val)
        end = time.time()
        t_nuc_val = round(end - st, 5)

    with open('../Results/times.json') as json_file:
        t = json.load(json_file)

    t[MODEL_NAME]["Softmax SE"]["uncertainty"] = t[MODEL_NAME]["Softmax SE"]["uncertainty"] + [t_soft_uncert]
    t[MODEL_NAME]["Softmax SE"]["calibration"] = t[MODEL_NAME]["Softmax SE"]["calibration"] + [t_soft_calib]
    t[MODEL_NAME]["MC Dropout"]["with calibration"] = t[MODEL_NAME]["MC Dropout"]["with calibration"] + [t_drop_calib]

    if not SAVE_OR_USE_SAVED_MODELS:
        t[MODEL_NAME]["Data Augmentation"]["preparation & calibration"] = \
            t[MODEL_NAME]["Data Augmentation"]["preparation & calibration"] + [t_da]
        t[MODEL_NAME]["ZIS"]["preparation & calibration"] = \
            t[MODEL_NAME]["ZIS"]["preparation & calibration"] + [t_ris]
        t[MODEL_NAME]["Bagging"]["preparation & calibration"] = \
            t[MODEL_NAME]["Bagging"]["preparation & calibration"] + [t_bag]
        t[MODEL_NAME]["NUC Training"]["preparation & uncertainty"] = \
            t[MODEL_NAME]["NUC Training"]["preparation & uncertainty"] + [t_nuc_tr]
        if MODEL_NAME != "CNN_cifar10_100":
            t[MODEL_NAME]["NUC Validation"]["preparation & uncertainty"] = \
                t[MODEL_NAME]["NUC Validation"]["preparation & uncertainty"] + [t_nuc_val]

    else:
        t[MODEL_NAME]["Data Augmentation"]["with calibration"] = \
            t[MODEL_NAME]["Data Augmentation"]["with calibration"] + [t_da]
        t[MODEL_NAME]["ZIS"]["with calibration"] = t[MODEL_NAME]["ZIS"]["with calibration"] + [t_ris]
        t[MODEL_NAME]["Bagging"]["with calibration"] = t[MODEL_NAME]["Bagging"]["with calibration"] + [t_bag]
        t[MODEL_NAME]["NUC Training"]["uncertainty"] = t[MODEL_NAME]["NUC Training"]["uncertainty"] + [t_nuc_tr]
        if MODEL_NAME != "CNN_cifar10_100":
            t[MODEL_NAME]["NUC Validation"]["uncertainty"] = t[MODEL_NAME]["NUC Validation"]["uncertainty"] + [t_nuc_val]

    with open('../Results/times.json', 'w') as json_file:
        json.dump(t, json_file, indent=4)

    y_pred_drop = MCEstimator.get_ensemble_prediction()
    y_pred_bag = BaEstimator.get_ensemble_prediction()
    y_pred_aug = DAEstimator.get_ensemble_prediction()
    y_pred_rand = RISEstimator.get_ensemble_prediction()

    uncert_mcdr_se = MCEstimator.uncertainties_shannon_entropy()
    uncert_mcdr_mi = MCEstimator.uncertainties_mutual_information()
    uncert_bag_se = BaEstimator.uncertainties_shannon_entropy()
    uncert_bag_mi = BaEstimator.uncertainties_mutual_information()
    uncert_rand_se = RISEstimator.uncertainties_shannon_entropy()
    uncert_rand_mi = RISEstimator.uncertainties_mutual_information()
    uncert_aug_se = DAEstimator.uncertainties_shannon_entropy()
    uncert_aug_mi = DAEstimator.uncertainties_mutual_information()

    cert_mcdr_se = MCEstimator.normalized_certainties_shannon_entropy()
    cert_mcdr_mi = MCEstimator.normalized_certainties_mutual_information()
    cert_bag_se = BaEstimator.normalized_certainties_shannon_entropy()
    cert_bag_mi = BaEstimator.normalized_certainties_mutual_information()
    cert_rand_se = RISEstimator.normalized_certainties_shannon_entropy()
    cert_rand_mi = RISEstimator.normalized_certainties_mutual_information()
    cert_aug_se = DAEstimator.normalized_certainties_shannon_entropy()
    cert_aug_mi = DAEstimator.normalized_certainties_mutual_information()

    methods_auroc_aupr = ["MCdrop SE", "MCdrop MI", "Bag SE", "Bag MI", "Rand SE", "Rand MI", "DataAug SE",
                          "DataAug MI", "Softmax"]
    methods_pre_spe_rec = ["MCD SE", "MCD MI", "Bag SE", "Bag MI", "ZIS SE", "ZIS MI", "DA SE", "DA MI", "Soft SE",
                           "NUC Tr", "NUC Va"]

    preds_auroc_aupr = [y_pred_drop, y_pred_drop, y_pred_bag, y_pred_bag, y_pred_rand, y_pred_rand, y_pred_aug,
                        y_pred_aug, y_pred]
    preds_pre_spe_rec = preds_auroc_aupr + [y_pred, y_pred]

    if MODEL_NAME == "CNN_cifar10_100":
        methods_pre_spe_rec.pop(-1)
        preds_pre_spe_rec.pop(-1)
        print(methods_pre_spe_rec)

    # normalize uncertainties between 0 and 1 to make the metrics' calculation more precise
    uncert_mcdr_se -= tf.reduce_min(uncert_mcdr_se)
    uncert_mcdr_mi -= tf.reduce_min(uncert_mcdr_mi)
    uncert_bag_se -= tf.reduce_min(uncert_bag_se)
    uncert_bag_mi -= tf.reduce_min(uncert_bag_mi)
    uncert_rand_se -= tf.reduce_min(uncert_rand_se)
    uncert_rand_mi -= tf.reduce_min(uncert_rand_mi)
    uncert_aug_se -= tf.reduce_min(uncert_aug_se)
    uncert_aug_mi -= tf.reduce_min(uncert_aug_mi)
    uncerts_auroc_aupr = [uncert_mcdr_se/tf.reduce_max(uncert_mcdr_se), uncert_mcdr_mi/tf.reduce_max(uncert_mcdr_mi),
                          uncert_bag_se/tf.reduce_max(uncert_bag_se), uncert_bag_mi/tf.reduce_max(uncert_bag_mi),
                          uncert_rand_se/tf.reduce_max(uncert_rand_se), uncert_rand_mi/tf.reduce_max(uncert_rand_mi),
                          uncert_aug_se/tf.reduce_max(uncert_aug_se), uncert_aug_mi/tf.reduce_max(uncert_aug_mi),
                          soft_ent_uncert_test/tf.reduce_max(soft_ent_uncert_test)]
    uncerts_auroc_aupr = [tf.clip_by_value(uncerts, 0, 1) for uncerts in uncerts_auroc_aupr]

    if MODEL_NAME != "CNN_cifar10_100":
        uncerts_pre_spe_rec = uncerts_auroc_aupr + [1-NUEstimator_on_train.certainties, 1-NUEstimator.certainties]
        certs_pre_spe_rec = [cert_mcdr_se, cert_mcdr_mi, cert_bag_se, cert_bag_mi, cert_rand_se, cert_rand_mi,
                             cert_aug_se, cert_aug_mi, softmax_entropy, NUEstimator_on_train.certainties,
                             NUEstimator.certainties]
    else:
        uncerts_pre_spe_rec = uncerts_auroc_aupr + [1 - NUEstimator_on_train.certainties]
        certs_pre_spe_rec = [cert_mcdr_se, cert_mcdr_mi, cert_bag_se, cert_bag_mi, cert_rand_se, cert_rand_mi,
                             cert_aug_se, cert_aug_mi, softmax_entropy, NUEstimator_on_train.certainties]

    uncerts_pre_spe_rec = [tf.clip_by_value(uncerts, 0, 1) for uncerts in uncerts_pre_spe_rec]


    # calculate AUROC and AUPR
    with open('../Results/auroc_aupr.json') as json_file:
        data = json.load(json_file)

    for m, uncert, pred in zip(methods_auroc_aupr, uncerts_auroc_aupr, preds_auroc_aupr):
        roc = auroc(lbls, pred, uncert)
        pr = aupr(lbls, pred, uncert)
        print(m)
        print("AUROC: ", roc)
        print("AUPR: ", pr, "\n")

        data[m][MODEL_NAME]["auroc"] = data[m][MODEL_NAME]["auroc"] + [roc.item()]
        data[m][MODEL_NAME]["aupr"] = data[m][MODEL_NAME]["aupr"] + [pr.item()]

    with open('../Results/auroc_aupr.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    # calculate Precision, Specificity, Recall
    for pred, uncert, cert, m in zip(preds_pre_spe_rec, uncerts_pre_spe_rec, certs_pre_spe_rec, methods_pre_spe_rec):
        with open('../Results/pre_spe_rec.json') as json_file:
            data = json.load(json_file)

        # use calibrated certainties
        pre, spe, rec = Evaluator(lbls, pred, cert).results(thresholds=list(np.linspace(0.6, 1, 20)))
        data[MODEL_NAME][m]["pre"]["calibrated"] = data[MODEL_NAME][m]["pre"]["calibrated"] + [list(pre)]
        data[MODEL_NAME][m]["spe"]["calibrated"] = data[MODEL_NAME][m]["spe"]["calibrated"] + [list(spe)]
        data[MODEL_NAME][m]["rec"]["calibrated"] = data[MODEL_NAME][m]["rec"]["calibrated"] + [list(rec)]

        # get values for roc- and pr-curves
        pre, spe, rec = calculate_metrics(lbls, pred, uncert)
        data[MODEL_NAME][m]["pre"]["curves"] = data[MODEL_NAME][m]["pre"]["curves"] + [list(pre)]
        data[MODEL_NAME][m]["spe"]["curves"] = data[MODEL_NAME][m]["spe"]["curves"] + [list(spe)]
        data[MODEL_NAME][m]["rec"]["curves"] = data[MODEL_NAME][m]["rec"]["curves"] + [list(rec)]

        with open('../Results/pre_spe_rec.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

    # calculate calibration error
    with open('../Results/eces.json') as json_file:
        data = json.load(json_file)

    for pred, cert, m in zip(preds_pre_spe_rec, certs_pre_spe_rec, methods_pre_spe_rec):
        ece = expected_calibration_error(lbls, pred, cert).numpy()
        data[MODEL_NAME][m] = data[MODEL_NAME][m] + [ece.item()]

    with open('../Results/eces.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)