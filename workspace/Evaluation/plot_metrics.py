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
THRESHOLDS = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95]

path_to_bagging_ens = ENSEMBLE_LOCATION + "/bagging/" + MODEL + "_" + DATA
path_to_dataAug_ens = ENSEMBLE_LOCATION + "/data_augmentation/" + MODEL + "_" + DATA
path_to_randInitShuffle_ens = ENSEMBLE_LOCATION + "/rand_initialization_shuffle/" + MODEL + "_" + DATA
path_uncertainty_model = "../models/classification/uncertainty_model/" + DATA + "/cp.ckpt"
path_uncertainty_model_on_train = "../models/classification/uncertainty_model/trained_on_traindata/" + DATA + "/cp.ckpt"


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

    def accuracy(self, groups):
        """
        ratio of the sum of correct uncertainties (TC+TU) to all predictions made by the model
        :param groups: True Uncertain (TU), True Certain (TC), False Uncertain (FU), False Certain (FC)
        :return: certainty accuracy
        """
        TU, TC, FU, FC = groups
        return (TU+TC)/(TU+TC+FC+FU)

    def results(self):
        accs, preci, speci, rec = [], [], [], []
        for tr in THRESHOLDS:
            gr = self.make_groups(tr)
            accs.append(self.accuracy(gr))
            preci.append(self.precision(gr))
            speci.append(self.specificity(gr))
            rec.append(self.recall(gr))
        return accs, preci, speci, rec


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

val = True
MCEstimator = MCDropoutEstimator(model, x_test, num_classes, T=50, xval=x_val, yval=y_val)
DAEstimator = DataAugmentationEns(x, y, x_test, num_classes, model_name=model_name,
                                  path_to_ensemble=path_to_dataAug_ens, num_members=NUM_MEMBERS,
                                  X_val=x_val, y_val=y_val, val=val)
RISEstimator = RandomInitShuffleEns(x, y, x_test, num_classes, model_name=model_name,
                                    path_to_ensemble=path_to_randInitShuffle_ens, num_members=NUM_MEMBERS,
                                    X_val=x_val, y_val=y_val, val=val)
BaEstimator = BaggingEns(x, y, x_test, num_classes, model_name=model_name, path_to_ensemble=path_to_bagging_ens,
                         num_members=NUM_MEMBERS, X_val=x_val, y_val=y_val, val=val)

if MODEL == "effnet":
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
soft_ent_uncert_val = tfd.Categorical(probs=model.predict(x_val, verbose=0)).entropy().numpy()
softmax_entropy = get_normalized_certainties(model.predict(x_val, verbose=0), y_val,
                                             soft_ent_uncert_val, soft_ent_uncert_test)
mcdr_se = MCEstimator.normalized_certainties_shannon_entropy()
mcdr_mi = MCEstimator.normalized_certainties_mutual_information()
bag_se = BaEstimator.normalized_certainties_shannon_entropy()
bag_mi = BaEstimator.normalized_certainties_mutual_information()
rand_se = RISEstimator.normalized_certainties_shannon_entropy()
rand_mi = RISEstimator.normalized_certainties_mutual_information()
aug_se = DAEstimator.normalized_certainties_shannon_entropy()
aug_mi = DAEstimator.normalized_certainties_mutual_information()

certainties = [mcdr_se, mcdr_mi, bag_se, bag_mi, rand_se, rand_mi,
               aug_se, aug_mi,
               NUEstimator.certainties,
               softmax_entropy]

results = [Evaluator(lbls, pred, certainty).results() for certainty, pred in zip(certainties, preds)]

plt.figure(figsize=(10, 10))
plt.suptitle(MODEL + " ---- " + DATA, fontsize=14)
plt.subplot(2, 2, 1)
for i, res in enumerate(results):
    subplot_evaluation(res[0], "Uncertainty Accuracy", methods[i])
plt.subplot(2, 2, 2)
for i, res in enumerate(results):
    subplot_evaluation(res[1], "Uncertainty Precision", methods[i])
plt.subplot(2, 2, 3)
for i, res in enumerate(results):
    subplot_evaluation(res[2], "Uncertainty Specificity", methods[i])
plt.subplot(2, 2, 4)
for i, res in enumerate(results):
    subplot_evaluation(res[3], "Uncertainty Recall", methods[i])
plt.legend(loc="lower right")
plt.show()
