import random
import json
import sys
import numpy as np
import tqdm
sys.path.append("/home/urz/hlichten")
from functools import partial
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from functions import get_data, CNN_transfer_learning, CNN
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.Ensemble import BaggingEns, DataAugmentationEns, RandomInitShuffleEns
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
import tensorflow_probability as tfp
tfd = tfp.distributions

STARTDATA = 100
NUM_IMAGES = 100
HYBRID = True
RUNS = 5

PATH_TO_PRETRAINED_CNN_10 = "../models/classification/CNN_cifar10_" + str(STARTDATA)
times_images_added = 10

file_name = "../Results/active_learning/" + str(STARTDATA) + "_" + str(NUM_IMAGES)
file_name = file_name + "_hybrid.json" if HYBRID else file_name + ".json"

xtrain, ytrain, xval, yval, xtest, ytest, _, xleft, yleft = get_data("cifar10", STARTDATA, active_learning=True)


def prepare_model(path=PATH_TO_PRETRAINED_CNN_10):
    model = tf.keras.models.load_model(path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def explorationP(features, uncert, indices_new_data, num_data):
    similarity = 0
    for f1 in [features[i] for i in indices_new_data]:
        for f2 in [features[i] for i in indices_new_data]:
            similarity += tf.reshape(tf.matmul(f1, f2), ()).numpy()
    sum_uncerts = tf.reduce_sum([uncert[i] for i in indices_new_data])
    print(indices_new_data)
    print(similarity / num_data)
    print(sum_uncerts)
    I = sum_uncerts - similarity / num_data
    return I


def calculate_similarity(features):
    sim_matrix = tf.linalg.matvec(features, features)
    sim = tf.reduce_mean(tf.reshape(sim_matrix, (-1)))
    return sim


def remove_dominated(batches, unc, sim):
    to_remove = []
    for count, (u, s) in tqdm.tqdm(enumerate(zip(unc, sim))):
        for u_, s_ in zip(unc, sim):
            if u < u_ and s > s_:
                to_remove.append(count)
                break
    for index in reversed(to_remove):
        del batches[index]
        del unc[index]
        del sim[index]
    return batches, unc, sim


class RetrainingEvaluator:

    def __init__(self):
        self.X_left, self.y_left = xleft, yleft
        self.X_train, self.y_train = xtrain, ytrain

    def retrain(self, model, num_data, uncert, just_div=False):

        if HYBRID:
            indices_new_data = self.hybrid_query(model, num_data, uncert, just_div)
        else:
            assert not just_div
            index = sorted(range(len(uncert)), key=uncert.__getitem__, reverse=True)
            indices_new_data = index[:num_data]

        self.X_train, self.y_train = list(self.X_train), list(self.y_train)
        self.X_left, self.y_left = list(self.X_left), list(self.y_left)

        for i in indices_new_data:
            self.X_train.append(self.X_left[i])
            self.y_train.append(self.y_left[i])

        self.X_left = [self.X_left[i] for i in range(len(self.X_left)) if i not in indices_new_data]
        self.y_left = [self.y_left[i] for i in range(len(self.y_left)) if i not in indices_new_data]

        self.X_left, self.y_left = tf.convert_to_tensor(self.X_left), tf.convert_to_tensor(self.y_left)
        self.X_train, self.y_train = tf.convert_to_tensor(self.X_train), tf.convert_to_tensor(self.y_train)

        # transfer learning
        # first step: weights of main model frozen
        model = CNN_transfer_learning()
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3, restore_best_weights=True)
        model.fit(self.X_train, self.y_train, validation_data=(xval, yval),
                  callbacks=[early_stop], verbose=1, epochs=1000, batch_size=128 if STARTDATA >= 10000 else 32)
        # if convergence: begin second step
        model.trainable = True
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15,
                                  restore_best_weights=True)
        rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)
        model.fit(self.X_train, self.y_train, validation_data=(xval, yval),
                  callbacks=[early_stop, rlrop], verbose=1, epochs=1000, batch_size=128 if STARTDATA >= 10000 else 32)
        loss, acc = model.evaluate(xtest, ytest, verbose=2)
        return acc, model

    def hybrid_query(self, model, num_data, uncert, just_div):
        output = model.layers[-3].output
        model_without_last_layer = tf.keras.Model(inputs=model.input, outputs=output)
        model_without_last_layer.compile()
        features = model_without_last_layer.predict(self.X_left)
        indices = [i for i in range(len(self.X_left))]

        ind = tf.random.shuffle(indices)
        batch_indices = [ind[j:j+num_data] for j in range(0, len(self.X_left), num_data) if j+num_data <= len(self.X_left)]
        sum_uncertainty, similarity = [], []
        for batch in tqdm.tqdm(batch_indices):
            if not just_div:
                sum_uncertainty.append(tf.reduce_sum([uncert[i] for i in batch]))
            similarity.append(calculate_similarity([features[i] for i in batch]))

        if just_div:
            most_diverse_batch = tf.argmax(similarity)
            new_indices = batch_indices[most_diverse_batch]
            return tf.reshape(new_indices, (-1)).numpy()

        plt.scatter(sum_uncertainty, similarity, label="Batches S")
        plt.xlabel("E(S)")
        plt.ylabel("R(S)")

        batch_indices, sum_uncertainty, similarity = remove_dominated(batch_indices, sum_uncertainty, similarity)
        plt.scatter(sum_uncertainty, similarity, c="red", label="Pareto Front")
        middle = sorted(range(len(similarity)), key=similarity.__getitem__)[len(similarity)//2]
        new_indices = batch_indices[middle]
        plt.scatter(sum_uncertainty[middle], similarity[middle], c="green", label="Auswahl")
        plt.legend(loc="lower left")
        plt.savefig("../plots/pareto_front.pdf")
        plt.show()
        return tf.reshape(new_indices, (-1)).numpy()


def retrain_with_ensemble(ensemble, metric):
    for _ in range(RUNS):
        model = prepare_model()
        retrainer = RetrainingEvaluator()

        with open(file_name) as json_file:
            data = json.load(json_file)

        for i in range(times_images_added):
            f = CNN if STARTDATA != 100 else CNN_transfer_learning
            uncertainty_estimator = ensemble(retrainer.X_left, num_classes=10, build_model_function=f,
                                             X_train=retrainer.X_train, y_train=retrainer.y_train,
                                             X_val=xval, y_val=yval)

            if metric == "SE":
                uncertainties = uncertainty_estimator.uncertainties_shannon_entropy()
                acc, model = retrainer.retrain(model, NUM_IMAGES, uncertainties)
                data[str(STARTDATA + (i+1)*NUM_IMAGES)]["Ensembles"][ensemble.__name__]["SE"] = \
                    data[str(STARTDATA+(i+1)*NUM_IMAGES)]["Ensembles"][ensemble.__name__]["SE"] + [acc]

            if metric == "MI":
                uncertainties = uncertainty_estimator.uncertainties_mutual_information()
                acc, model = retrainer.retrain(model, NUM_IMAGES, uncertainties)
                data[str(STARTDATA + (i+1) * NUM_IMAGES)]["Ensembles"][ensemble.__name__]["MI"] = \
                    data[str(STARTDATA+(i+1)*NUM_IMAGES)]["Ensembles"][ensemble.__name__]["MI"] + [acc]

        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)


def retrain_with_MCdrop(metric):
    for _ in range(RUNS):
        model = prepare_model()
        retrainer = RetrainingEvaluator()
        with open(file_name) as json_file:
            data = json.load(json_file)

        for i in range(times_images_added):
            clone_model = prepare_model()
            clone_model.set_weights(model.get_weights())
            uncertainty_estimator = MCDropoutEstimator(clone_model, retrainer.X_left, 10, 50)

            if metric == "SE":
                uncertainties = uncertainty_estimator.uncertainties_shannon_entropy()
                acc, model = retrainer.retrain(model, NUM_IMAGES, uncertainties)
                data[str(STARTDATA + (i + 1) * NUM_IMAGES)]["MC_drop"]["SE"] = \
                    data[str(STARTDATA + (i + 1) * NUM_IMAGES)]["MC_drop"]["SE"] + [acc]

            if metric == "MI":
                uncertainties = uncertainty_estimator.uncertainties_mutual_information()
                acc, model = retrainer.retrain(model, NUM_IMAGES, uncertainties)
                data[str(STARTDATA + (i + 1) * NUM_IMAGES)]["MC_drop"]["MI"] = \
                    data[str(STARTDATA + (i + 1) * NUM_IMAGES)]["MC_drop"]["MI"] + [acc]

        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)


def retrain_with_nuc(train_data):

    for _ in range(RUNS):
        model = prepare_model()
        retrainer = RetrainingEvaluator()
        if train_data:
            x_train = retrainer.X_train
            y_train = retrainer.y_train
            x_val, y_val = xval, yval
            k = 25
        else:
            k = 5
            x_train = xval[:int(4*len(xval) / 5)]
            y_train = yval[:int(4*len(yval) / 5)]
            x_val = xval[int(4*len(xval) / 5):]
            y_val = yval[int(4*len(yval) / 5):]

        for i in range(times_images_added):
            uncertainty_estimator = NeighborhoodUncertaintyClassifier(model, x_train, y_train, x_val, y_val,
                                                                      retrainer.X_left, k=k)
            acc, model = retrainer.retrain(model, NUM_IMAGES, 1 - uncertainty_estimator.certainties)
            method = "NUC Tr" if train_data else "NUC"

            with open(file_name) as json_file:
                data = json.load(json_file)
                data[str(STARTDATA + (i+1)*NUM_IMAGES)][method] = \
                    data[str(STARTDATA + (i+1)*NUM_IMAGES)][method] + [acc]
            with open(file_name, 'w') as json_file:
                json.dump(data, json_file, indent=4)


def retrain_with_softmax_entropy():
    for _ in range(RUNS):
        model = prepare_model()
        retrainer = RetrainingEvaluator()

        for i in range(times_images_added):
            out = model.predict(retrainer.X_left)
            shannon_entropy = tfd.Categorical(probs=out).entropy().numpy()
            acc, model = retrainer.retrain(model, NUM_IMAGES, shannon_entropy)

            with open(file_name) as json_file:
                data = json.load(json_file)
                data[str(STARTDATA + (i+1)*NUM_IMAGES)]["softmax_entropy"] = \
                    data[str(STARTDATA + (i+1)*NUM_IMAGES)]["softmax_entropy"] + [acc]
            with open(file_name, 'w') as json_file:
                json.dump(data, json_file, indent=4)


def retrain_with_random_data():
    for _ in range(RUNS):
        model = prepare_model()
        retrainer = RetrainingEvaluator()

        for i in range(times_images_added):
            uncertainties = [random.random() for _ in retrainer.y_left]
            acc, model = retrainer.retrain(model, NUM_IMAGES, uncertainties)

            with open(file_name) as json_file:
                data = json.load(json_file)
                data[str(STARTDATA + (i + 1) * NUM_IMAGES)]["random"] = \
                    data[str(STARTDATA + (i + 1) * NUM_IMAGES)]["random"] + [acc]
            with open(file_name, 'w') as json_file:
                json.dump(data, json_file, indent=4)


def retrain_with_diverse_imgs():
    for _ in range(RUNS):
        model = prepare_model()
        retrainer = RetrainingEvaluator()

        for i in range(times_images_added):
            uncertainties = []
            acc, model = retrainer.retrain(model, NUM_IMAGES, uncertainties, just_div=True)

            with open(file_name) as json_file:
                data = json.load(json_file)
                data[str(STARTDATA + (i + 1) * NUM_IMAGES)]["just_divers"] = \
                    data[str(STARTDATA + (i + 1) * NUM_IMAGES)]["just_divers"] + [acc]
            with open(file_name, 'w') as json_file:
                json.dump(data, json_file, indent=4)



prepare_model().evaluate(xtest, ytest)
# check whether the classes are balanced in train dataset
print([list(tf.argmax(ytrain, axis=-1)).count(i) for i in range(10)])

retrain_with_nuc(train_data=False)
retrain_with_softmax_entropy()
retrain_with_random_data()
#retrain_with_diverse_imgs()
retrain_with_nuc(train_data=True)
retrain_with_MCdrop("MI")
retrain_with_MCdrop("SE")
retrain_with_ensemble(DataAugmentationEns, "SE")
retrain_with_ensemble(DataAugmentationEns, "MI")
retrain_with_ensemble(BaggingEns, "SE")
retrain_with_ensemble(BaggingEns, "MI")

