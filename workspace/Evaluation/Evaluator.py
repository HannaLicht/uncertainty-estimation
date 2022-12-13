import matplotlib.pyplot as plt
import tensorflow as tf
import statistics
import sys
sys.path.append("/home/urz/hlichten")
sys.path.append("/home/hanna/Schreibtisch/Ingenieurinformatik VW/Igenieurinformatik/BA/uncertainty-estimation/workspace")
print(sys.path)

from functions import CNN, get_train_and_test_data
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.Ensemble import ENSEMBLE_LOCATION, BaggingEns, DataAugmentationEns, RandomInitShuffleEns
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
import tensorflow_probability as tfp
tfd = tfp.distributions

DATA = "cifar10_100"
MODEL = "CNN"
CHECKPOINT_PATH = "models/classification/" + MODEL + "_" + DATA + "/cp.ckpt"
NUM_MEMBERS = 5

path_to_bagging_ens = ENSEMBLE_LOCATION + "/bagging/" + MODEL + "_" + DATA
path_to_dataAug_ens = ENSEMBLE_LOCATION + "/data_augmentation/" + MODEL + "_" + DATA
path_to_randInitShuffle_ens = ENSEMBLE_LOCATION + "/rand_initialization_shuffle/" + MODEL + "_" + DATA
path_uncertainty_model = "models/classification/uncertainty_model/" + DATA + "/cp.ckpt"


class Evaluator:

    def __init__(self, lbls: list, preds: list, certainties):
        self.correct = (lbls == preds)
        self.certainties = certainties

    def auroc(self):
        m = tf.keras.metrics.AUC(curve='ROC')
        m.update_state(self.correct, self.certainties)
        return m.result().numpy()

    def aupr(self):
        m = tf.keras.metrics.AUC(curve="PR")
        m.update_state(self.correct, self.certainties)
        return m.result().numpy()

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
        indicates how much of the positive outputs are correctly labeled as positive:
        ratio of incorrect predictions that are uncertain (TU) to all the incorrect predictions (TU+FC)
        :param groups: True Uncertain (TU), True Certain (TC), False Uncertain (FU), False Certain (FC)
        :return: sensitivity ratio
        """
        TU = groups[0]
        FC = groups[3]
        return TU/(TU+FC)

    def specificity(self, groups):
        """
        true negative rate:
        ratio of correct predictions that are certain (TC) to all the right predictions (TC+FU)
        :param groups: True Uncertain (TU), True Certain (TC), False Uncertain (FU), False Certain (FC)
        :return: specificity ratio
        """
        TC = groups[1]
        FU = groups[2]
        return TC/(TC + FU)

    def precision(self, groups):
        """
        positive predictive value:
        ratio of uncertain and incorrect (TU) predictions to all uncertain predictions (TU+FU)
        :param groups: True Uncertain (TU), True Certain (TC), False Uncertain (FU), False Certain (FC)
        :return: precision ratio
        """
        TU = groups[0]
        FU = groups[2]
        try:
            result = TU/(TU+FU)
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

    def results(self, thresholds):
        accs, preci, speci, rec = [], [], [], []
        for tr in thresholds:
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
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #thr = THRESHOLDS
    while values[0] is None:
        values = values[1:]
        quantiles = quantiles[1:]
    plt.plot(quantiles, values, label=method)
    plt.xlabel("Certainty limit in quantiles")
    plt.ylabel(eval_metric)


model = CNN(classes=100 if DATA == "cifar100" else 10)
model.load_weights(CHECKPOINT_PATH)

x, y, x_test, y_test, num_classes = get_train_and_test_data("cifar10" if DATA == "cifar10_1000" or
                                                                         DATA == "cifar10_10000" or
                                                                         DATA == "cifar10_100" else DATA)
if DATA == "cifar10_1000":
    x, y = x[:1000], y[:1000]
elif DATA == "cifar10_10000":
    x, y = x[:10000], y[:10000]
elif DATA == "cifar10_100":
    x, y = x[:100], y[:100]

_, acc = model.evaluate(x, y)
print("Accuracy on train dataset: ", acc)
_, acc = model.evaluate(x_test, y_test)
print("Accuracy on test dataset: ", acc)

model_name = MODEL + "_cifar10" if DATA == "cifar10_1000" or DATA == "cifar10_10000" or DATA == "cifar10_100" \
    else MODEL + "_" + DATA

MCEstimator = MCDropoutEstimator(model, x_test, num_classes, T=50)
DAEstimator = DataAugmentationEns(x, y, x_test, num_classes, model_name=model_name,
                                  path_to_ensemble=path_to_dataAug_ens, num_members=NUM_MEMBERS,
                                  X_test=x_test, y_test=y_test)
RISEstimator = RandomInitShuffleEns(x, y, x_test, num_classes, model_name=model_name,
                                    path_to_ensemble=path_to_randInitShuffle_ens, num_members=NUM_MEMBERS,
                                    X_test=x_test, y_test=y_test)
BaEstimator = BaggingEns(x, y, x_test, num_classes, model_name=model_name, path_to_ensemble=path_to_bagging_ens,
                         num_members=NUM_MEMBERS, X_test=x_test, y_test=y_test)
NUEstimator = NeighborhoodUncertaintyClassifier(model, x, y, x_test, y_test,
                                                path_uncertainty_model=path_uncertainty_model)
methods = ["MCdrop SE", "MCdrop MI", "Bag SE", "Bag MI", "Rand SE", "Rand MI",
           "DataAug SE", "DataAug MI", "NUC", "Softmax"]

lbls = tf.math.argmax(y_test, axis=-1)
y_pred = tf.math.argmax(model.predict(x_test, verbose=0), axis=-1)
y_pred_drop = MCEstimator.get_ensemble_prediction()
y_pred_bag = BaEstimator.get_ensemble_prediction()
y_pred_aug = DAEstimator.get_ensemble_prediction()
y_pred_rand = RISEstimator.get_ensemble_prediction()
preds = [y_pred_drop, y_pred_drop, y_pred_bag, y_pred_bag, y_pred_rand, y_pred_rand,
         y_pred_aug, y_pred_aug,
         y_pred, y_pred]

softmax_entropy = tfd.Categorical(probs=model.predict(x_test, verbose=0)).entropy().numpy()
print("Max SE single softmax: ", tf.math.reduce_max(softmax_entropy).numpy())

mcdr_se = MCEstimator.uncertainties_shannon_entropy()
print("Max SE mc dropout: ", tf.math.reduce_max(mcdr_se).numpy())
mcdr_mi = MCEstimator.uncertainties_mutual_information()
print("Max MI mc dropout: ", tf.math.reduce_max(mcdr_mi).numpy())
bag_se = BaEstimator.uncertainties_shannon_entropy()
print("Max SE bagging: ", tf.math.reduce_max(bag_se).numpy())
bag_mi = BaEstimator.uncertainties_mutual_information()
print("Max MI bagging: ", tf.math.reduce_max(bag_mi).numpy())
rand_se = RISEstimator.uncertainties_shannon_entropy()
print("Max SE rand. initialization & data shuffle: ", tf.math.reduce_max(rand_se).numpy())
rand_mi = RISEstimator.uncertainties_mutual_information()
print("Max MI rand. initialization & data shuffle: ", tf.math.reduce_max(rand_mi).numpy())
aug_se = DAEstimator.uncertainties_shannon_entropy()
print("Max SE data augmentation: ", tf.math.reduce_max(aug_se).numpy())
aug_mi = DAEstimator.uncertainties_mutual_information()
print("Max MI data augmentation: ", tf.math.reduce_max(aug_mi).numpy())

# make certainties between 0 and 1
certainties = [1 - mcdr_se/tf.reduce_max(mcdr_se), 1 - mcdr_mi/tf.reduce_max(mcdr_mi),
               1 - bag_se/tf.reduce_max(bag_se), 1 - bag_mi/tf.reduce_max(bag_mi),
               1 - rand_se/tf.reduce_max(rand_se), 1 - rand_mi/tf.reduce_max(rand_mi),
               1 - aug_se/tf.reduce_max(aug_se), 1 - aug_mi/tf.reduce_max(aug_mi),
               NUEstimator.certainties, 1 - softmax_entropy/tf.reduce_max(softmax_entropy)
               ]
#thresholds = [statistics.quantiles(cert, n=10) for cert in certainties]
#results = [Evaluator(lbls, y_pred, certainty).results(thr) for certainty, thr in zip(certainties, thresholds)]
print(methods)
print([Evaluator(lbls, pred, cert).auroc() for cert, pred in zip(certainties, preds)])
print([Evaluator(lbls, pred, cert).aupr() for cert, pred in zip(certainties, preds)])

'''
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
'''

