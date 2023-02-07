import random
import json
import sys

import numpy as np

sys.path.append("/home/urz/hlichten")
print(sys.path)
from functools import partial
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from functions import get_train_and_test_data, CNN, adjust_lightness
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.Ensemble import BaggingEns, DataAugmentationEns, RandomInitShuffleEns
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
import tensorflow_probability as tfp
tfd = tfp.distributions

STARTDATA = 100
NUM_IMAGES = 10
RUNS = 6
PATH_TO_PRETRAINED_CNN_10 = "../models/classification/retrain/CNN_cifar10_" + str(STARTDATA) + "/cp.ckpt"
PATH_TO_PRETRAINED_CNN_100 = "../models/classification/retrain/CNN_cifar100/cp.ckpt"


times_images_added = 10 if STARTDATA == 10000 else 9

(xleft, yleft), (xtest, ytest) = tf.keras.datasets.cifar10.load_data()
xleft = xleft.reshape(-1, 32, 32, 3) / 255.0
xtest = xtest.reshape(-1, 32, 32, 3) / 255.0

xtrain, ytrain = xleft[:STARTDATA], yleft[:STARTDATA]
xleft, yleft = xleft[STARTDATA:], yleft[STARTDATA:]


def mean(l):
    return sum(l)/len(l)


def train_base_model(checkpoint_path_old, checkpoint_path_new, X_train, y_train, X_test, y_test):
    """
    :param checkpoint_path_old: path of pretrained model CNN_cifar100
    :param checkpoint_path_new: path to save model further trained on the 'amount_of_data' first train data of
                                CNN_cifar10
    :return:
    """
    model = CNN(classes=10)
    model.load_weights(checkpoint_path_old)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15,
                               restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)
    X_train, y_train = tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_new,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     verbose=1)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              callbacks=[early_stop, cp_callback, rlrop], verbose=1, epochs=1000)


def prepare_model(path=PATH_TO_PRETRAINED_CNN_10):
    model = CNN(classes=10)
    model.load_weights(path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class RetrainingEvaluator:

    def __init__(self):
        self.X_left, self.y_left = xleft, tf.keras.utils.to_categorical(yleft.reshape((-1)), 10)
        self.X_train, self.y_train = xtrain, tf.keras.utils.to_categorical(ytrain.reshape((-1)), 10)
        self.X_test, self.y_test = xtest, tf.keras.utils.to_categorical(ytest.reshape((-1)), 10)

    def retrain(self, model, num_data, cert):
        self.X_train, self.y_train = list(self.X_train), list(self.y_train)
        index = sorted(range(len(cert)), key=cert.__getitem__, reverse=False)

        for i in range(num_data):
            self.X_train.append(self.X_left[index[i]])
            self.y_train.append(self.y_left[index[i]])

        self.X_train, self.y_train = tf.convert_to_tensor(self.X_train), tf.convert_to_tensor(self.y_train)
        self.X_left = tf.convert_to_tensor([self.X_left[ind] for ind in index[num_data:]])
        self.y_left = tf.convert_to_tensor([self.y_left[ind] for ind in index[num_data:]])

        early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=15,
                                   restore_best_weights=True)
        rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test),
                  callbacks=[early_stop, rlrop], verbose=1, epochs=1000, batch_size=128 if STARTDATA >= 1000 else 32)
        loss, acc = model.evaluate(self.X_test, self.y_test, verbose=2)
        return acc, model


def retrain_with_ensemble(ensemble, metric):
    for _ in range(RUNS):
        model = prepare_model()
        retrainer = RetrainingEvaluator()

        with open('../Results/retrain.json') as json_file:
            data = json.load(json_file)

        for i in range(times_images_added):
            uncertainty_estimator = ensemble(retrainer.X_left, num_classes=10, model_name="CNN_cifar10",
                                             X_train=retrainer.X_train, y_train=retrainer.y_train,
                                             X_val=retrainer.X_test, y_val=retrainer.y_test)

            if metric == "SE":
                certainties = -uncertainty_estimator.uncertainties_shannon_entropy()
                acc, model = retrainer.retrain(model, NUM_IMAGES, certainties)
                data[str(STARTDATA)][str(STARTDATA + (i+1)*NUM_IMAGES)]["Ensembles"][ensemble.__name__]["SE"] = \
                    data[str(STARTDATA)][str(STARTDATA+(i+1)*NUM_IMAGES)]["Ensembles"][ensemble.__name__]["SE"] + [acc]

            if metric == "MI":
                certainties = -uncertainty_estimator.uncertainties_mutual_information()
                acc, model = retrainer.retrain(model, NUM_IMAGES, certainties)
                data[str(STARTDATA)][str(STARTDATA + (i+1) * NUM_IMAGES)]["Ensembles"][ensemble.__name__]["MI"] = \
                    data[str(STARTDATA)][str(STARTDATA+(i+1)*NUM_IMAGES)]["Ensembles"][ensemble.__name__]["MI"] + [acc]

        with open('../Results/retrain.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)


def retrain_with_MCdrop(metric):
    for _ in range(RUNS):
        model = prepare_model()
        retrainer = RetrainingEvaluator()
        with open('../Results/retrain.json') as json_file:
            data = json.load(json_file)

        for i in range(times_images_added):
            clone_model = prepare_model()
            clone_model.set_weights(model.get_weights())
            uncertainty_estimator = MCDropoutEstimator(clone_model, retrainer.X_left, 10, 50)

            if metric == "SE":
                certainties = -uncertainty_estimator.uncertainties_shannon_entropy()
                acc, model = retrainer.retrain(model, NUM_IMAGES, certainties)
                data[str(STARTDATA)][str(STARTDATA + (i + 1) * NUM_IMAGES)]["MC_drop"]["SE"] = \
                    data[str(STARTDATA)][str(STARTDATA + (i + 1) * NUM_IMAGES)]["MC_drop"]["SE"] + [acc]

            if metric == "MI":
                certainties = -uncertainty_estimator.uncertainties_mutual_information()
                acc, model = retrainer.retrain(model, NUM_IMAGES, certainties)
                data[str(STARTDATA)][str(STARTDATA + (i + 1) * NUM_IMAGES)]["MC_drop"]["MI"] = \
                    data[str(STARTDATA)][str(STARTDATA + (i + 1) * NUM_IMAGES)]["MC_drop"]["MI"] + [acc]

        with open('../Results/retrain.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)


def retrain_with_nuc():

    for _ in range(RUNS):
        model = prepare_model()
        retrainer = RetrainingEvaluator()
        xtrain = retrainer.X_test[:int(4*len(retrainer.X_test) / 5)]
        ytrain = retrainer.y_test[:int(4*len(retrainer.y_test) / 5)]
        xval = retrainer.X_test[int(4*len(retrainer.X_test) / 5):]
        yval = retrainer.y_test[int(4*len(retrainer.y_test) / 5):]

        for i in range(times_images_added):
            uncertainty_estimator = NeighborhoodUncertaintyClassifier(model, xtrain, ytrain, xval, yval,
                                                                      retrainer.X_left)
            acc, model = retrainer.retrain(model, NUM_IMAGES, uncertainty_estimator.certainties)

            with open('../Results/retrain.json') as json_file:
                data = json.load(json_file)
                data[str(STARTDATA)][str(STARTDATA + (i+1)*NUM_IMAGES)]["NUC"] = \
                    data[str(STARTDATA)][str(STARTDATA + (i+1)*NUM_IMAGES)]["NUC"] + [acc]
            with open('../Results/retrain.json', 'w') as json_file:
                json.dump(data, json_file, indent=4)


def retrain_with_softmax_entropy():
    for _ in range(RUNS):
        model = prepare_model()
        retrainer = RetrainingEvaluator()

        for i in range(times_images_added):
            out = model.predict(retrainer.X_left)
            shannon_entropy = tfd.Categorical(probs=out).entropy().numpy()
            certainties = 1 - shannon_entropy
            acc, model = retrainer.retrain(model, NUM_IMAGES, certainties)

            with open('../Results/retrain.json') as json_file:
                data = json.load(json_file)
                data[str(STARTDATA)][str(STARTDATA + (i+1)*NUM_IMAGES)]["softmax_entropy"] = \
                    data[str(STARTDATA)][str(STARTDATA + (i+1)*NUM_IMAGES)]["softmax_entropy"] + [acc]
            with open('../Results/retrain.json', 'w') as json_file:
                json.dump(data, json_file, indent=4)


def retrain_with_random_data():
    for _ in range(RUNS):
        model = prepare_model()
        retrainer = RetrainingEvaluator()

        for i in range(times_images_added):
            certainties = [random.random() for _ in retrainer.y_left]
            acc, model = retrainer.retrain(model, NUM_IMAGES, certainties)

            with open('../Results/retrain.json') as json_file:
                data = json.load(json_file)
                data[str(STARTDATA)][str(STARTDATA + (i + 1) * NUM_IMAGES)]["random"] = \
                    data[str(STARTDATA)][str(STARTDATA + (i + 1) * NUM_IMAGES)]["random"] + [acc]
            with open('../Results/retrain.json', 'w') as json_file:
                json.dump(data, json_file, indent=4)


# comment if model already trained on STARTDATA of CNN_cifar10 training data
#train_base_model(PATH_TO_PRETRAINED_CNN_100, PATH_TO_PRETRAINED_CNN_10,
 #                xtrain, tf.keras.utils.to_categorical(ytrain.reshape((-1)), 10),
  #               xtest, tf.keras.utils.to_categorical(ytest.reshape((-1)), 10))

prepare_model().evaluate(xtest, tf.keras.utils.to_categorical(ytest.reshape((-1)), 10))
# check whether the classes are balanced in train dataset
print([list(ytrain).count(i) for i in range(10)])

retrain_with_nuc()
retrain_with_ensemble(DataAugmentationEns, "SE")
retrain_with_ensemble(DataAugmentationEns, "MI")
retrain_with_ensemble(RandomInitShuffleEns, "SE")
retrain_with_ensemble(RandomInitShuffleEns, "MI")
retrain_with_MCdrop("SE")
retrain_with_MCdrop("MI")
retrain_with_softmax_entropy()
retrain_with_random_data()
retrain_with_ensemble(BaggingEns, "SE")
retrain_with_ensemble(BaggingEns, "MI")

with open('../Results/retrain.json') as json_file:
    data = json.load(json_file)
    data = data[str(STARTDATA)]

numbers = [STARTDATA + i*NUM_IMAGES for i in range(times_images_added+1)]
rand = [mean(data[str(imgs)]["random"])*100 for imgs in numbers]
softmax = [mean(data[str(imgs)]["softmax_entropy"])*100 for imgs in numbers]
mc_se = [mean(data[str(imgs)]["MC_drop"]["SE"])*100 for imgs in numbers]
mc_mi = [mean(data[str(imgs)]["MC_drop"]["MI"])*100 for imgs in numbers]
bag_se = [mean(data[str(imgs)]["Ensembles"]["BaggingEns"]["SE"])*100 for imgs in numbers]
bag_mi = [mean(data[str(imgs)]["Ensembles"]["BaggingEns"]["MI"])*100 for imgs in numbers]
aug_se = [mean(data[str(imgs)]["Ensembles"]["DataAugmentationEns"]["SE"])*100 for imgs in numbers]
aug_mi = [mean(data[str(imgs)]["Ensembles"]["DataAugmentationEns"]["MI"])*100 for imgs in numbers]
ris_se = [mean(data[str(imgs)]["Ensembles"]["RandomInitShuffleEns"]["SE"])*100 for imgs in numbers]
ris_mi = [mean(data[str(imgs)]["Ensembles"]["RandomInitShuffleEns"]["MI"])*100 for imgs in numbers]
nuc = [mean(data[str(imgs)]["NUC"])*100 for imgs in numbers]


plt.figure(figsize=(8, 4.5))

if STARTDATA == 100:
    methods_to_show = [rand, nuc, mc_mi, aug_mi, mc_se, bag_mi, softmax, ris_mi, bag_se, ris_se, aug_se]
    labels = ["Zufall", "NUC",  "MC Dropout MI", "Data Aug. MI", "MC Dropout SE", "Bagging MI", "Softmax SE", "ZIS MI",
              "Bagging SE", "ZIS SE", "Data Aug. SE"]
    colors = ["black", adjust_lightness("green", 1.3), adjust_lightness("green", 1.6), "yellowgreen",
              adjust_lightness("olive", 1.5), adjust_lightness("yellow", 0.9), "gold",  "orange",
              adjust_lightness("coral", 0.9), adjust_lightness("red", 1.1), adjust_lightness("red", 0.8)
              ]
    plt.ylim(33.6, 51.8)
    #plt.xticks([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

elif STARTDATA == 1000 and NUM_IMAGES == 1000:
    methods_to_show = [rand, nuc, mc_mi, ris_mi, softmax, bag_mi, aug_mi, mc_se, bag_se, ris_se, aug_se]
    labels = ["Zufall", "NUC", "MC Dropout MI", "ZIS MI", "Softmax SE", "Bagging MI", "Data Aug. MI", "MC Dropout SE",
              "Bagging SE", "ZIS SE", "Data Aug. SE"]
    colors = ["black", adjust_lightness("green", 1.3), adjust_lightness("green", 1.6), "yellowgreen",
              adjust_lightness("olive", 1.5), adjust_lightness("yellow", 0.9), "gold", "orange",
              adjust_lightness("coral", 0.9), adjust_lightness("red", 1.1), adjust_lightness("red", 0.8)
              ]
    plt.ylim(54.5, 67.8)

elif STARTDATA == 1000 and NUM_IMAGES == 100:
    methods_to_show = [rand, nuc, softmax, mc_mi, ris_mi, bag_mi, aug_mi, mc_se, ris_se, aug_se, bag_se]
    labels = ["Zufall", "NUC", "Softmax SE", "MC Dropout MI", "ZIS MI", "Bagging MI", "Data Aug. MI", "MC Dropout SE",
              "ZIS SE", "Data Aug. SE", "Bagging SE"]
    colors = ["black", adjust_lightness("green", 1.3), adjust_lightness("green", 1.6), "yellowgreen",
              adjust_lightness("olive", 1.5), adjust_lightness("yellow", 0.9), "gold", "orange",
              adjust_lightness("coral", 0.9), adjust_lightness("red", 1.1), adjust_lightness("red", 0.8)
              ]
    plt.ylim(51.5, 57.2)

else:
    methods_to_show = [rand, mc_mi, nuc, softmax, mc_se, bag_mi, aug_mi, ris_mi, bag_se, ris_se, aug_se]
    labels = ["Zufall", "MC Dropout MI", "NUC", "Softmax SE", "MC Dropout SE", "Bagging MI", "Data Aug. MI", "ZIS MI",
              "Bagging SE", "ZIS SE", "Data Aug. SE"]
    colors = ["black", adjust_lightness("green", 1.3), adjust_lightness("green", 1.6), "yellowgreen",
              adjust_lightness("olive", 1.5), adjust_lightness("yellow", 0.9), "gold", "orange",
              adjust_lightness("coral", 0.9), adjust_lightness("red", 1.1), adjust_lightness("red", 0.8)
              ]
    plt.ylim(67.6, 71.6)

for i, (method, l, c) in enumerate(zip(methods_to_show, labels, colors)):
    plt.plot(numbers, method, "--" if l == "Zufall" else "-", label=l, color=c, zorder=20 if l == "Zufall" else 15-i)
    if l == "Zufall":
        plt.plot(numbers, method, label=" ", color="white", zorder=0)
plt.xlabel("Anzahl gelabelter Bilder")
plt.ylabel("Testaccuracy in %")

plt.legend(bbox_to_anchor=(1, 1))
plt.subplots_adjust(left=0.1, right=0.75, bottom=0.12, top=0.95, wspace=0.3, hspace=0.35)
plt.savefig("../plots/active_learning_" + str(STARTDATA) + "_" + str(NUM_IMAGES) + ".pdf")

plt.show()