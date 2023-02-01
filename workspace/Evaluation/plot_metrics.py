import re

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from uncertainty.calibration_classification import get_normalized_certainties
import sys
sys.path.append("/home/urz/hlichten")
from functions import CNN, get_train_and_test_data, split_validation_from_train, build_effnet
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.Ensemble import ENSEMBLE_LOCATION, BaggingEns, DataAugmentationEns, RandomInitShuffleEns
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
import tensorflow_probability as tfp
tfd = tfp.distributions

DATA = "cifar10"
MODEL_NAME = "CNN_cifar10"
NUM_MEMBERS = 5
THRESHOLDS = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95, .99]

path_to_bagging_ens = ENSEMBLE_LOCATION + "/bagging/" + MODEL_NAME
path_to_dataAug_ens = ENSEMBLE_LOCATION + "/data_augmentation/" + MODEL_NAME
path_to_randInitShuffle_ens = ENSEMBLE_LOCATION + "/rand_initialization_shuffle/" + MODEL_NAME
model_path = "../models/classification/" + MODEL_NAME + "/cp.ckpt"
path_uncertainty_model = "../models/classification/uncertainty_model/" + MODEL_NAME + "/cp.ckpt"
path_uncertainty_model_on_train = "../models/classification/uncertainty_model/trained_on_traindata/" + MODEL_NAME + "/cp.ckpt"


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
        TC = groups[1]
        FU = groups[2]
        return TC/(TC+FU)

    def specificity(self, groups):
        """
        true negative rate:
        ratio of incorrect predictions that are uncertain (TU) to all the incorrect predictions (TU+FC)
        :param groups: True Uncertain (TU), True Certain (TC), False Uncertain (FU), False Certain (FC)
        :return: specificity ratio
        """
        TU = groups[0]
        FC = groups[3]
        return TU/(TU + FC)

    def precision(self, groups):
        """
        positive predictive value:
        ratio of certain and correct (TC) predictions to all certain predictions (TC+FC)
        :param groups: True Uncertain (TU), True Certain (TC), False Uncertain (FU), False Certain (FC)
        :return: precision ratio
        """
        TC = groups[1]
        FC = groups[3]
        try:
            result = TC/(TC+FC)
            return result
        except ZeroDivisionError:
            return None

    '''def accuracy(self, groups):
        """
        ratio of the sum of correct uncertainties (TC+TU) to all predictions made by the model
        :param groups: True Uncertain (TU), True Certain (TC), False Uncertain (FU), False Certain (FC)
        :return: certainty accuracy
        """
        TU, TC, FU, FC = groups
        return (TU+TC)/(TU+TC+FC+FU)'''

    def results(self):
        preci, speci, rec = [], [], []
        for tr in THRESHOLDS:
            gr = self.make_groups(tr)
            preci.append(self.precision(gr))
            speci.append(self.specificity(gr))
            rec.append(self.recall(gr))
        return preci, speci, rec


def optimal_certainties(lbls, preds):
    cur_max_cert = len(lbls) - 1
    cur_min_cert = 0
    certs = []
    for lbl, pred in zip(lbls, preds):
        if lbl == pred:
            certs.append(cur_max_cert)
            cur_max_cert = cur_max_cert - 1
        else:
            certs.append(cur_min_cert)
            cur_min_cert = cur_min_cert + 1
    return certs


def subplot_evaluation(values, eval_metric: str, method: str):
    thr = THRESHOLDS
    while values[0] is None:
        values = values[1:]
        thr = thr[1:]
    plt.plot(thr, values, label=method)
    plt.xlabel("Konfidenzlimit in Prozent")
    plt.ylabel(eval_metric)


def decor_plot(ax):
    plt.ylim(-0.02, 1.1)
    plt.xlim(-0.02, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.arrow(x=0, y=0, dx=1.05, dy=0, head_width=0.02, head_length=0.06, fc='black', ec='black', lw=0.2)
    ax.arrow(x=0, y=0, dx=0, dy=1.05, head_width=0.02, head_length=0.06, fc='black', ec='black', lw=0.2)


def plot_roc_and_pr_curves(results_metrics, labels):
    plt.figure(figsize=(9, 3.5))
    #rec_good, pr_good = tf.concat((rec_good, [0.]), axis=0), tf.concat((pr_good, [1.]), axis=0)

    ax = plt.subplot(1, 2, 1)
    plt.title("ROC-Kurven")
    for pre, fpr, rec in results_metrics:
        plt.plot(fpr, rec)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    decor_plot(ax)

    ax = plt.subplot(1, 2, 2)
    plt.title("PR-Kurven")
    for (pre, fpr, rec), lbl in zip(results_metrics, labels):
        plt.plot(rec, pre, label=lbl)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    decor_plot(ax)

    plt.legend(bbox_to_anchor=(1, 0.35))
    plt.savefig("../plots/roc_and_pr_curve.pdf")
    plt.show()


def plot_pre_spe_rec(results_metrics, labels):
    model_name = MODEL_NAME.replace('_', ' ')
    plt.figure(figsize=(10, 2.8))
    plt.suptitle(model_name, fontsize=14)
    plt.subplot(1, 3, 1)
    for i, res in enumerate(results_metrics):
        subplot_evaluation(res[0], "Uncertainty Precision", labels[i])
    plt.subplot(1, 3, 2)
    for i, res in enumerate(results_metrics):
        subplot_evaluation(res[1], "Uncertainty Specificity", labels[i])
    plt.subplot(1, 3, 3)
    for i, res in enumerate(results_metrics):
        subplot_evaluation(res[2], "Uncertainty Recall", labels[i])
    plt.legend(loc="lower left")
    plt.savefig("../plots/pre_spe_rec.pdf")
    plt.show()


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

val = True
MCEstimator = MCDropoutEstimator(model, x_test, num_classes, T=50, xval=x_val, yval=y_val)
#DAEstimator = DataAugmentationEns(x_test, num_classes, model_name=MODEL_NAME,
 #                                 path_to_ensemble=path_to_dataAug_ens, num_members=NUM_MEMBERS,
  #                                X_val=x_val, y_val=y_val, val=val)
RISEstimator = RandomInitShuffleEns(x_test, num_classes, model_name=MODEL_NAME,
                                    path_to_ensemble=path_to_randInitShuffle_ens, num_members=NUM_MEMBERS,
                                    X_val=x_val, y_val=y_val, val=val)
#BaEstimator = BaggingEns(x_test, num_classes, model_name=MODEL_NAME, path_to_ensemble=path_to_bagging_ens,
 #                        num_members=NUM_MEMBERS, X_val=x_val, y_val=y_val, val=val)

if MODEL_NAME == "effnet":
    nuc_xtrain, nuc_ytrain, nuc_xval, nuc_yval = split_validation_from_train(x_val, y_val, num_classes,
                                                                             num_imgs_per_class=2)
else:
    nuc_xtrain, nuc_ytrain = x_val[:int(4*len(x_val) / 5)], y_val[:int(4*len(y_val) / 5)]
    nuc_xval, nuc_yval = x_val[int(4*len(x_val) / 5):], y_val[int(4*len(y_val) / 5):]

NUEstimator = NeighborhoodUncertaintyClassifier(model, nuc_xtrain, nuc_ytrain, nuc_xval, nuc_yval, x_test,
                                                path_uncertainty_model=path_uncertainty_model)
#NUEstimator_on_train = NeighborhoodUncertaintyClassifier(model, x, y, x_val, y_val, x_test,
 #                                                        path_uncertainty_model=path_uncertainty_model_on_train)

methods = ["MCdrop SE", "MCdrop MI",
           #"Bag SE", "Bag MI",
           "Rand SE", "Rand MI",
           #"DataAug SE", #"DataAug MI",
           #"NUC_train",
           "NUC_valid",
           #"Softmax"
           ]

y_pred_drop = MCEstimator.get_ensemble_prediction()
#y_pred_bag = BaEstimator.get_ensemble_prediction()
#y_pred_aug = DAEstimator.get_ensemble_prediction()
y_pred_rand = RISEstimator.get_ensemble_prediction()
preds = [y_pred_drop, y_pred_drop,# y_pred_bag, y_pred_bag,
         y_pred_rand, y_pred_rand,
         #y_pred_aug, y_pred_aug,
         y_pred#, y_pred, y_pred
         ]

#soft_ent_uncert_test = tfd.Categorical(probs=model.predict(x_test, verbose=0)).entropy().numpy()
#soft_ent_uncert_val = tfd.Categorical(probs=model.predict(x_val, verbose=0)).entropy().numpy()
#softmax_entropy = get_normalized_certainties(model.predict(x_val, verbose=0), y_val,
#                                             soft_ent_uncert_val, soft_ent_uncert_test)
mcdr_se = MCEstimator.normalized_certainties_shannon_entropy()
mcdr_mi = MCEstimator.normalized_certainties_mutual_information()
#bag_se = BaEstimator.normalized_certainties_shannon_entropy()
#bag_mi = BaEstimator.normalized_certainties_mutual_information()
rand_se = RISEstimator.normalized_certainties_shannon_entropy()
rand_mi = RISEstimator.normalized_certainties_mutual_information()
#aug_se = DAEstimator.normalized_certainties_shannon_entropy()
#aug_mi = DAEstimator.normalized_certainties_mutual_information()

mcdr_se_u = MCEstimator.uncertainties_shannon_entropy()
mcdr_mi_u = MCEstimator.uncertainties_mutual_information()
#bag_se_u = BaEstimator.uncertainties_shannon_entropy()
#bag_mi_u = BaEstimator.uncertainties_mutual_information()
rand_se_u = RISEstimator.uncertainties_shannon_entropy()
rand_mi_u = RISEstimator.uncertainties_mutual_information()
#aug_se_u = DAEstimator.uncertainties_shannon_entropy()
#aug_mi_u = DAEstimator.uncertainties_mutual_information()

certainties = [mcdr_se, mcdr_mi,# bag_se, bag_mi,
               rand_se,
               rand_mi,
               #aug_se,
               #aug_mi,
               #NUEstimator_on_train.certainties,
               NUEstimator.certainties,
               #softmax_entropy
               ]

uncertainties = [mcdr_se_u/tf.reduce_max(mcdr_se_u), mcdr_mi_u/tf.reduce_max(mcdr_mi_u),
               # bag_se_u//tf.reduce_max(bag_se_u), bag_mi_u//tf.reduce_max(bag_mi_u),
               rand_se_u/tf.reduce_max(rand_se_u), rand_mi_u//tf.reduce_max(rand_mi_u),
               #aug_se_u//tf.reduce_max(aug_se_u), aug_mi_u//tf.reduce_max(aug_mi_u),
               #1-NUEstimator_on_train.certainties,
               1-NUEstimator.certainties,
               #soft_ent_uncert_test/tf.reduce_max(soft_ent_uncert_test)
               ]

#results = [Evaluator(lbls, pred, certainty).results() for certainty, pred in zip(certainties, preds)]
#plot_pre_spe_rec(results, methods)

results = []

for pred, uncert in zip(preds, uncertainties):
    pr = tf.metrics.Precision(thresholds=list(np.linspace(0, 1, 200)))
    rec = tf.metrics.Recall(thresholds=list(np.linspace(0, 1, 200)))
    fp = tf.metrics.FalsePositives(thresholds=list(np.linspace(0, 1, 200)))
    tn = tf.metrics.TrueNegatives(thresholds=list(np.linspace(0, 1, 200)))

    incorrect = (lbls != pred)
    pr.update_state(incorrect, uncert)
    rec.update_state(incorrect, uncert)
    fp.update_state(incorrect, uncert)
    tn.update_state(incorrect, uncert)

    fpr = (fp.result() / (fp.result() + tn.result())).numpy()
    results.append([pr.result().numpy(), fpr, rec.result().numpy()])

plot_roc_and_pr_curves(results, methods)
