import json
import re
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import sys
sys.path.append("/home/urz/hlichten")
from functions import get_data, split_validation_from_train, build_effnet, CNN_transfer_learning, adjust_lightness, \
    COLORS, CNN
from uncertainty.Ensemble import DataAugmentationEns, RandomInitShuffleEns, BaggingEns
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
from uncertainty.calibration_classification import get_normalized_certainties
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

RUNS = 5
DATA = "cars196"


if DATA == "cifar10" or DATA == "mnist":
    num_used_classes = 5
elif DATA == "cifar100":
    num_used_classes = 50
elif DATA == "cars196":
    num_used_classes = 100
else:
    raise NotImplementedError

if DATA == "cars196":
    model_name = "effnetb3"
    shape = (-1, 300, 300, 3)
else:
    model_name = "CNN_cifar" + str(num_used_classes)
    shape = (-1, 32, 32, 3)

model_path = "../models/classification/" + model_name if model_name != "effnetb3" else \
    "../models/classification/effnetb3_ood"

methods = ["MCD SE", "Bag SE", "DA SE", "Soft SE", "MCD MI", "Bag MI", "DA MI", "NUC Tr", "NUC Va", "Max Soft"]
colors = [COLORS[m] for m in methods[:-1]] + ["black"]
thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]


def split_up_some_classes(x, y):
    x_new, y_new, unknown_classes = [], [], []
    for img, lbl in zip(x, y):
        index = tf.argmax(lbl)
        lbl = list(lbl)

        if DATA == "cifar10" or DATA == "mnist":
            # leave out airplane, bird, frog, horse and ship in case of cifar10
            leave_out = index == 0 or index == 2 or index == 6 or index == 7 or index == 8
            # make one-hot-vector for 5 classes
            for i in [8, 7, 6, 2, 0]:
                lbl.pop(i)
        elif DATA == "cifar100":
            # leave out last 50 classes
            leave_out = index > 49
            for i in reversed(range(50, 100)):
                lbl.pop(i)
        elif DATA == "cars196":
            # leave out last 96 classes
            leave_out = index > 99
            for i in reversed(range(100, 196)):
                lbl.pop(i)
        else:
            raise NotImplementedError

        if leave_out:
            unknown_classes.append(img)
        else:
            x_new.append(img)
            y_new.append(lbl)

    x_new = tf.reshape(x_new, shape)
    y_new = tf.reshape(y_new, (-1, num_used_classes))

    return x_new, y_new, unknown_classes


# get train, val and test datasets without frogs
xtrain, ytrain, xval, yval, xtest, ytest, _ = get_data(DATA)
train_images, train_labels, unknown_train = split_up_some_classes(xtrain, ytrain)
val_images, val_labels, unknown_val = split_up_some_classes(xval, yval)
test_images, test_labels, unknown_test = split_up_some_classes(xtest, ytest)
ood = tf.concat([unknown_train, unknown_val, unknown_test], axis=0)
if DATA == "cars196":
    ood = ood[:8000]
ood_and_test_data = tf.concat([ood, test_images], axis=0)
if DATA == "cars196":
    ood_and_test_data = ood_and_test_data[:12000]
print(ood.shape)
print(ood_and_test_data.shape)
ood = tf.reshape(ood, shape)
ood_and_test_data = tf.reshape(ood_and_test_data, shape)

plt.figure(figsize=(10, 10))
for i in range(9):
    img = ood[i]
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.axis("off")
plt.show()

if DATA == "cifar10":
    build_model_function = CNN_transfer_learning
elif DATA == "cars196":
    build_model_function = build_effnet
else:
    build_model_function = CNN

try:
    model = tf.keras.models.load_model(model_path)

except:
    print("no model found for ood detection")
    model = build_model_function(num_used_classes)

    if build_model_function == build_effnet:
        early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=15,
                                   restore_best_weights=True)
        early_stop_transfer = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=3,
                                            restore_best_weights=True)
    else:
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15,
                                   restore_best_weights=True)
        early_stop_transfer = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3,
                                            restore_best_weights=True)

    if DATA != "cifar100":
        # transfer learning
        model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=1000,
                  batch_size=128, callbacks=[early_stop_transfer])
        # if convergence: begin second step
        model.trainable = True
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3 if DATA != "cars196" else 1e-4)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)
    model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=1000,
              batch_size=128, callbacks=[early_stop, rlrop])
    model.save(model_path)

model.evaluate(test_images, test_labels, verbose=2)

for _ in range(RUNS):
    model = tf.keras.models.load_model(model_path)

    if DATA == "cars196":
        xtrain_nuc_val, ytrain_nuc_val, xval_nuc_val, yval_nuc_val = \
            split_validation_from_train(val_images, val_labels, num_used_classes, num_imgs_per_class=2)
    else:
        xtrain_nuc_val, ytrain_nuc_val = val_images[:int(4 * len(val_images) / 5)], val_labels[:int(4 * len(val_images) / 5)]
        xval_nuc_val, yval_nuc_val = val_images[int(4 * len(val_images) / 5):], val_labels[int(4 * len(val_images) / 5):]

    MCEstimator = MCDropoutEstimator(model, ood_and_test_data, num_used_classes, T=50)
    DAEstimator = DataAugmentationEns(ood_and_test_data, num_used_classes, X_train=train_images, y_train=train_labels,
                                      X_val=val_images, y_val=val_labels, val=False, build_model_function=build_model_function)
    BaEstimator = BaggingEns(ood_and_test_data, num_used_classes, X_train=train_images, y_train=train_labels,
                             X_val=val_images, y_val=val_labels, val=False, build_model_function=build_model_function)
    nuc_train = NeighborhoodUncertaintyClassifier(model, train_images, train_labels, val_images, val_labels, ood_and_test_data)
    nuc_val = NeighborhoodUncertaintyClassifier(model, xtrain_nuc_val, ytrain_nuc_val, xval_nuc_val, yval_nuc_val, ood_and_test_data)

    max_soft = tf.reduce_max(model.predict(ood_and_test_data, verbose=0), axis=-1).numpy()

    soft_ent_uncert_ood = tfd.Categorical(probs=model.predict(ood_and_test_data, verbose=0)).entropy().numpy()
    '''soft_ent_uncert_val = tfd.Categorical(probs=model.predict(val_images, verbose=0)).entropy().numpy()
    softmax_entropy = get_normalized_certainties(model.predict(val_images, verbose=0), val_labels,
                                                 soft_ent_uncert_val, soft_ent_uncert_ood)
    mcdr_se = MCEstimator.normalized_certainties_shannon_entropy()
    mcdr_mi = MCEstimator.normalized_certainties_mutual_information()
    bag_se = BaEstimator.normalized_certainties_shannon_entropy()
    bag_mi = BaEstimator.normalized_certainties_mutual_information()
    aug_se = DAEstimator.normalized_certainties_shannon_entropy()
    aug_mi = DAEstimator.normalized_certainties_mutual_information()

    certainties = [mcdr_se, mcdr_mi,
                   bag_se, bag_mi, aug_se, aug_mi,
                   nuc_train.certainties, nuc_val.certainties, 
                   softmax_entropy, max_soft]'''

    '''for certs, method in zip(certainties, methods):
        TU, FC = [], []
        for thr in thresholds:
            count = 0
            for cert in certs:
                if cert < thr:
                    count = count+1
            TU.append(count)
            FC.append(len(certs)-count)
        recall = tf.divide(TU, tf.add(TU, FC)).numpy()
        with open('../Results/recalls_ood.json') as json_file:
            data = json.load(json_file)
            for i in range(len(thresholds)):
                data[DATA][method][i] = data[DATA][method][i] + [recall[i].item()]
        with open('../Results/recalls_ood.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)'''

    y_pred = tf.math.argmax(model.predict(ood_and_test_data), axis=-1).numpy()
    y_pred_drop = MCEstimator.get_ensemble_prediction()
    y_pred_bag = BaEstimator.get_ensemble_prediction()
    y_pred_aug = DAEstimator.get_ensemble_prediction()

    uncert_mcdr_se = MCEstimator.uncertainties_shannon_entropy()
    uncert_mcdr_mi = MCEstimator.uncertainties_mutual_information()
    uncert_bag_se = BaEstimator.uncertainties_shannon_entropy()
    uncert_bag_mi = BaEstimator.uncertainties_mutual_information()
    uncert_aug_se = DAEstimator.uncertainties_shannon_entropy()
    uncert_aug_mi = DAEstimator.uncertainties_mutual_information()

    preds= [y_pred_drop, y_pred_drop, y_pred_bag, y_pred_bag, y_pred_aug, y_pred_aug, y_pred, y_pred,
            y_pred, y_pred]
    methods = ["MCD SE", "MCD MI", "Bag SE", "Bag MI", "DA SE", "DA MI", "NUC Tr", "NUC Va", "Soft SE", "Max Soft"]

    # normalize uncertainties between 0 and 1 to make the metrics' calculation more precise
    uncert_mcdr_se -= tf.reduce_min(uncert_mcdr_se)
    uncert_mcdr_mi -= tf.reduce_min(uncert_mcdr_mi)
    uncert_bag_se -= tf.reduce_min(uncert_bag_se)
    uncert_bag_mi -= tf.reduce_min(uncert_bag_mi)
    uncert_aug_se -= tf.reduce_min(uncert_aug_se)
    uncert_aug_mi -= tf.reduce_min(uncert_aug_mi)
    uncerts = [uncert_mcdr_se / tf.reduce_max(uncert_mcdr_se), uncert_mcdr_mi / tf.reduce_max(uncert_mcdr_mi),
               uncert_bag_se / tf.reduce_max(uncert_bag_se), uncert_bag_mi / tf.reduce_max(uncert_bag_mi),
               uncert_aug_se / tf.reduce_max(uncert_aug_se), uncert_aug_mi / tf.reduce_max(uncert_aug_mi),
               1 - nuc_train.certainties, 1 - nuc_val.certainties,
               soft_ent_uncert_ood / tf.reduce_max(soft_ent_uncert_ood), 1 - max_soft]
    uncerts = [tf.clip_by_value(uncert, 0, 1) for uncert in uncerts]

    lbls_test = tf.argmax(test_labels, axis=-1)
    incorrect_ood_samples = [True for _ in range(len(ood))]
    incorrect_ood_samples = tf.reshape(incorrect_ood_samples, (-1))

    with open('../Results/ood_auroc_aupr.json') as json_file:
        data = json.load(json_file)

    for method, uncert, pred in zip(methods, uncerts, preds):

        incorrect_known_domain = (lbls_test != pred[len(ood):])
        incorrect_known_domain = tf.reshape(incorrect_known_domain, (-1))
        incorrect = tf.concat([incorrect_ood_samples, incorrect_known_domain], axis=0)
        assert len(incorrect) == len(pred)

        m = tf.keras.metrics.AUC(curve='ROC')
        m.update_state(incorrect, uncert)
        auroc = m.result().numpy()
        print(auroc)

        m = tf.keras.metrics.AUC(curve="PR")
        m.update_state(incorrect, uncert)
        aupr = m.result().numpy()
        print(aupr)

        data[DATA][method]["auroc"] = data[DATA][method]["auroc"] + [auroc.item()]
        data[DATA][method]["aupr"] = data[DATA][method]["aupr"] + [aupr.item()]

    with open('../Results/ood_auroc_aupr.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)


'''def subplot_evaluation(values, v, eval_metric: str, method: str, c):
    thr, values, v = thresholds[2:], values[2:], v[2:]
    l, = plt.plot(thr, values, label=method, color=c, linewidth=1., zorder=1, linestyle="--" if c == "black" else "-")
    plt.fill_between(thr, values + v, values - v, color=c, alpha=0.1, zorder=0)

    plt.xlabel("Certainty-Schwellenwert")
    plt.ylabel(eval_metric)
    plt.xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    plt.grid(visible=True, color="gainsboro", linestyle='dashed', zorder=0)

    return l


def plot_results(results, vari, labels, indices_best, y_min, y_max):
    plt.figure(figsize=(10, 3.5))
    name = "Recall"
    loc = "lower right"

    ax = plt.subplot(1, 3, 1)
    plt.ylim(y_min, y_max)
    ax.set_axisbelow(True)
    for res, v, lbl, c in zip(results[:4], vari[:4], labels[:4], colors[:4]):
        subplot_evaluation(res, v, name, lbl, c)
    subplot_evaluation(results[-1], vari[-1], name, labels[-1], colors[-1])
    plt.legend(loc=loc)

    ax = plt.subplot(1, 3, 2)
    plt.ylim(y_min, y_max)
    ax.set_axisbelow(True)
    for res, v, lbl, c in zip(results[4:], vari[4:], labels[4:], colors[4:]):
        subplot_evaluation(res, v, name, lbl, c)
    plt.legend(loc=loc)

    ax = plt.subplot(1, 3, 3)
    if DATA != "cars196":
        plt.ylim(y_min, y_max)
    else:
        plt.ylim(0.74, 1.01)
    ax.set_axisbelow(True)
    best_results = [results[i] for i in indices_best]
    best_vars = [vari[i] for i in indices_best]
    best_colors = [colors[i] for i in indices_best]
    best_lbl = [labels[i] for i in indices_best]
    for res, v, lbl, c in zip(best_results, best_vars, best_lbl, best_colors):
        subplot_evaluation(res, v, name, lbl, c)
    plt.legend(loc=loc)

    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.97, wspace=0.3, hspace=0.35)
    plt.savefig("../plots/ood_" + DATA + ".pdf")
    plt.show()


def plot_recalls(results, vari, labels):
    if DATA == "cars196":
        indices_rec = [0, 4, 8]
        min_rec, max_rec = 0.15, 1.02

    elif DATA == "cifar100":
        indices_rec = [0, 4, 8]
        min_rec, max_rec = 0.71, 1.01

    else:
        indices_rec = [0, 1, 4, 5, 8]
        min_rec, max_rec = 0.14, 1.02

    plot_results(results, vari, labels, indices_rec, min_rec, max_rec)


with open('../Results/recalls_ood.json') as json_file:
    data = json.load(json_file)

plt.figure(figsize=(10, 4))
data = data[DATA]
means = tf.reduce_mean([data[m] for m in methods], axis=-1)
stddevs = tf.math.reduce_std([data[m] for m in methods], axis=-1)
plot_recalls(means, stddevs, methods)
'''