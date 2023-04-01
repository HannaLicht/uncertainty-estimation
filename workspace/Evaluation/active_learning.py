import random
import json
import time
import tqdm
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from functions import get_data, CNN_transfer_learning, CNN
from Uncertainty.MC_Dropout import MCDropoutEstimator
from Uncertainty.Ensemble import BaggingEns, DataAugmentationEns, RandomInitShuffleEns
from Uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
import tensorflow_probability as tfp
tfd = tfp.distributions

""" 
Version of query method: U - choose based on max. uncertainty 
                         H - hybrid query (uncertainty & diversity)
                         L - long training (based on max. uncertainty, add 100 images till 25000 train images)
                         S - choose based on max. uncertainty from just 10000 unlabeled images
"""
VERSION = "U"

startdata = 1000        # size of initial train data set
num_images = 1000       # batch size
runs = 5

PATH_TO_PRETRAINED_CNN_10 = "../Models/classification/CNN_cifar10_" + str(startdata)
times_images_added = 10
xtrain, ytrain, xval, yval, xtest, ytest, _, xleft, yleft = get_data("cifar10", startdata, active_learning=True)
file_name = "../Results/active_learning/"

if VERSION == "L":
    assert num_images == 100 and startdata == 100
    times_images_added = 249
    file_name = file_name + "long_training.json"
    xval = tf.concat([xval, xleft[:4900]], axis=0)
    yval = tf.concat([yval, yleft[:4900]], axis=0)
    xleft, yleft = xleft[4900:], yleft[4900:]

elif VERSION == "H":
    file_name = file_name + str(startdata) + "_" + str(num_images) + "_hybrid.json"

elif VERSION == "S":
    assert num_images == 100 and startdata == 100
    file_name = file_name + str(startdata) + "_" + str(num_images) + "_smallU.json"
    xleft = xleft[:10000]
    yleft = yleft[:10000]

else:
    file_name = file_name + str(startdata) + "_" + str(num_images) + ".json"


def prepare_model(path=PATH_TO_PRETRAINED_CNN_10):
    model = tf.keras.models.load_model(path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def calculate_similarity(features):
    sim_matrix = tf.linalg.matvec(features, features)
    sim = tf.reduce_mean(tf.reshape(sim_matrix, (-1))).numpy()
    return sim


def calculate_similarity_search_algorthm(features, new_feature):
    if len(features) > 0:
        dot_products = tf.matmul(features, tf.reshape(new_feature, (-1, 1)))
        sim = tf.reduce_sum(dot_products)
    else:
        sim = 0
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


def remove_dominated_search_algorithm(unc, sim):
    to_remove = []
    for count, (u, s) in tqdm.tqdm(enumerate(zip(unc, sim))):
        for u_, s_ in zip(unc, sim):
            if u[1] < u_[1] and s[1] > s_[1]:
                to_remove.append(count)
                break
    for index in reversed(to_remove):
        del unc[index]
        del sim[index]
    return unc, sim


class RetrainingEvaluator:

    def __init__(self):
        self.X_left, self.y_left = xleft, yleft
        self.X_train, self.y_train = xtrain, ytrain

    def retrain(self, model, num_data, uncert, just_div=False):

        if VERSION == "H":
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
                  callbacks=[early_stop], verbose=1, epochs=1000, batch_size=128 if startdata >= 10000 else 32)

        # if convergence: begin second step
        model.trainable = True
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15,
                                   restore_best_weights=True)
        rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)
        model.fit(self.X_train, self.y_train, validation_data=(xval, yval),
                  callbacks=[early_stop, rlrop], verbose=1, epochs=1000, batch_size=128 if startdata >= 10000 else 32)
        loss, acc = model.evaluate(xtest, ytest, verbose=2)

        return acc, model

    def hybrid_query(self, model, num_data, uncert, just_div):
        output = model.layers[-3].output
        model_without_last_layer = tf.keras.Model(inputs=model.input, outputs=output)
        model_without_last_layer.compile()
        features = tf.math.softmax(model_without_last_layer.predict(self.X_left), axis=-1)

        indices = [i for i in range(len(self.X_left))]
        batch_indices = []
        for _ in range(10):
            ind = tf.random.shuffle(indices)
            batch_indices = batch_indices + [ind[j:j+num_data] for j in range(0, len(self.X_left), num_data)
                                             if j+num_data <= len(self.X_left)]
        sum_uncertainty, similarity = [], []
        for batch in tqdm.tqdm(batch_indices):
            if not just_div:
                sum_uncertainty.append(tf.reduce_sum([uncert[i] for i in batch]).numpy())
            similarity.append(calculate_similarity([features[i] for i in batch]))

        if just_div:
            most_diverse_batch = tf.argmax(similarity)
            new_indices = batch_indices[most_diverse_batch]
            return tf.reshape(new_indices, (-1)).numpy()

        batch_indices, sum_uncertainty, similarity = remove_dominated(batch_indices, sum_uncertainty, similarity)
        new_indices = random.sample(batch_indices, 1)

        return tf.reshape(new_indices, (-1)).numpy()

    def hybrid_query_search_algorithm(self, model, num_data, uncert, just_div):
            output = model.layers[-3].output
            model_without_last_layer = tf.keras.Model(inputs=model.input, outputs=output)
            model_without_last_layer.compile()
            features = tf.math.softmax(model_without_last_layer.predict(self.X_left), axis=-1)

            new_indices = []
            start = time.time()
            for _ in tqdm.tqdm(range(num_data)):
                uncertainty = []
                similarity = []
                for ind in range(len(self.X_left)):
                    if ind in new_indices:
                        continue
                    uncertainty.append([ind, uncert[ind]])
                    similarity.append([ind, calculate_similarity_search_algorthm(
                        [features[i] for i in new_indices], features[ind])])
                if just_div:
                    most_diverse_batch = tf.argmax(similarity, axis=0)[1]
                    new_indices.append(similarity[most_diverse_batch][0])
                else:
                    sum_uncertainty, similarity = remove_dominated_search_algorithm(uncertainty, similarity)
                    new_indices.append(tf.reshape(tf.cast(random.sample(sum_uncertainty, 1), tf.int32), (-1))[0])
                    new_indices = tf.reshape(new_indices, (-1))
            end = time.time()
            print("------ RUNTIME ------ : " + str(end-start))

            return new_indices.numpy()


def retrain_with_ensemble(ensemble, metric):
    for _ in range(runs):
        model = prepare_model()
        retrainer = RetrainingEvaluator()

        with open(file_name) as json_file:
            data = json.load(json_file)

        for i in range(times_images_added):
            uncertainty_estimator = ensemble(retrainer.X_left, num_classes=10,
                                             build_model_function=CNN_transfer_learning,
                                             X_train=retrainer.X_train, y_train=retrainer.y_train,
                                             X_val=xval, y_val=yval)

            if metric == "PE":
                uncertainties = uncertainty_estimator.uncertainties_shannon_entropy()
                acc, model = retrainer.retrain(model, num_images, uncertainties)
                data[str(startdata + (i+1)*num_images)]["Ensembles"][ensemble.__name__]["PE"] = \
                    data[str(startdata+(i+1)*num_images)]["Ensembles"][ensemble.__name__]["PE"] + [acc]

            if metric == "MI":
                uncertainties = uncertainty_estimator.uncertainties_mutual_information()
                acc, model = retrainer.retrain(model, num_images, uncertainties)
                data[str(startdata + (i+1) * num_images)]["Ensembles"][ensemble.__name__]["MI"] = \
                    data[str(startdata+(i+1)*num_images)]["Ensembles"][ensemble.__name__]["MI"] + [acc]

        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)


def retrain_with_MCdrop(metric):
    for _ in range(runs):
        model = prepare_model()
        retrainer = RetrainingEvaluator()
        with open(file_name) as json_file:
            data = json.load(json_file)

        for i in range(times_images_added):
            clone_model = prepare_model()
            clone_model.set_weights(model.get_weights())
            uncertainty_estimator = MCDropoutEstimator(clone_model, retrainer.X_left, 10, 50)

            if metric == "PE":
                uncertainties = uncertainty_estimator.uncertainties_shannon_entropy()
                acc, model = retrainer.retrain(model, num_images, uncertainties)
                data[str(startdata + (i + 1) * num_images)]["MC_drop"]["PE"] = \
                    data[str(startdata + (i + 1) * num_images)]["MC_drop"]["PE"] + [acc]

            if metric == "MI":
                uncertainties = uncertainty_estimator.uncertainties_mutual_information()
                acc, model = retrainer.retrain(model, num_images, uncertainties)
                if VERSION == "L":
                    data["MCD MI"] = data["MCD MI"] + [acc]
                else:
                    data[str(startdata + (i + 1) * num_images)]["MC_drop"]["MI"] = \
                        data[str(startdata + (i + 1) * num_images)]["MC_drop"]["MI"] + [acc]

        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)


def retrain_with_nuc(train_data):

    for _ in range(runs):
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
            acc, model = retrainer.retrain(model, num_images, 1 - uncertainty_estimator.certainties)
            method = "NUC Tr" if train_data else "NUC Va"

            with open(file_name) as json_file:
                data = json.load(json_file)
                if VERSION == "L" and not train_data:
                    data[method] = data[method] + [acc]
                else:
                    data[str(startdata + (i+1)*num_images)][method] = \
                        data[str(startdata + (i+1)*num_images)][method] + [acc]
            with open(file_name, 'w') as json_file:
                json.dump(data, json_file, indent=4)


def retrain_with_softmax_entropy():
    for _ in range(runs):
        model = prepare_model()
        retrainer = RetrainingEvaluator()

        for i in range(times_images_added):
            out = model.predict(retrainer.X_left)
            shannon_entropy = tfd.Categorical(probs=out).entropy().numpy()
            acc, model = retrainer.retrain(model, num_images, shannon_entropy)

            with open(file_name) as json_file:
                data = json.load(json_file)
                data[str(startdata + (i+1)*num_images)]["softmax_entropy"] = \
                    data[str(startdata + (i+1)*num_images)]["softmax_entropy"] + [acc]
            with open(file_name, 'w') as json_file:
                json.dump(data, json_file, indent=4)


def retrain_with_random_data():
    for _ in range(runs):
        model = prepare_model()
        retrainer = RetrainingEvaluator()

        for i in range(times_images_added):
            uncertainties = [random.random() for _ in retrainer.y_left]
            acc, model = retrainer.retrain(model, num_images, uncertainties)

            with open(file_name) as json_file:
                data = json.load(json_file)
                if VERSION == "L":
                    data["random"] = data["random"] + [acc]
                else:
                    data[str(startdata + (i + 1) * num_images)]["random"] = \
                        data[str(startdata + (i + 1) * num_images)]["random"] + [acc]
            with open(file_name, 'w') as json_file:
                json.dump(data, json_file, indent=4)


def retrain_with_diverse_imgs():
    for _ in range(runs):
        model = prepare_model()
        retrainer = RetrainingEvaluator()

        for i in range(times_images_added):
            uncertainties = [.0 for _ in retrainer.X_left]
            acc, model = retrainer.retrain(model, num_images, uncertainties, just_div=True)

            with open(file_name) as json_file:
                data = json.load(json_file)
                data[str(startdata + (i + 1) * num_images)]["just_divers"] = \
                    data[str(startdata + (i + 1) * num_images)]["just_divers"] + [acc]

            with open(file_name, 'w') as json_file:
                json.dump(data, json_file, indent=4)


prepare_model().evaluate(xtest, ytest)
# check whether classes are balanced in train dataset
print([list(tf.argmax(ytrain, axis=-1)).count(i) for i in range(10)])

if VERSION == "L":
    retrain_with_random_data()
    retrain_with_nuc(train_data=False)
    retrain_with_MCdrop("MI")
    retrain_with_ensemble(DataAugmentationEns, "MI")
    quit()

retrain_with_diverse_imgs() if VERSION == "H" else retrain_with_random_data()
retrain_with_softmax_entropy()
retrain_with_nuc(train_data=False)
retrain_with_nuc(train_data=True)
retrain_with_MCdrop("MI")
retrain_with_MCdrop("PE")
retrain_with_ensemble(DataAugmentationEns, "PE")
retrain_with_ensemble(DataAugmentationEns, "MI")
retrain_with_ensemble(BaggingEns, "PE")
retrain_with_ensemble(BaggingEns, "MI")

