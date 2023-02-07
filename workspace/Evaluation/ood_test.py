import json
import re
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import sys
sys.path.append("/home/urz/hlichten")
from functions import get_train_and_test_data, split_validation_from_train, build_effnet, CNN, adjust_lightness
from uncertainty.Ensemble import DataAugmentationEns, RandomInitShuffleEns, BaggingEns
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
from uncertainty.calibration_classification import get_normalized_certainties
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

RUNS = 1
DATA = "cifar100"


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
elif DATA == "mnist":
    model_name = "CNN_mnist5"
    shape = (-1, 28, 28, 1)
else:
    model_name = "CNN_cifar" + str(num_used_classes)
    shape = (-1, 32, 32, 3)

model_path = "../models/classification/" + model_name + "/cp.ckpt" if model_name != "effnetb3" else \
    "../models/classification/effnetb3_ood/cp.ckpt"

methods = [
    "MC Drop. SE", "MC Drop. MI",
    "ZIS SE", "ZIS MI", "Bagging SE", "Bagging MI", "Data Aug. SE", "Data Aug. MI",
    "NUC Train.", "NUC Valid.",
    "Softmax SE", "Max Softmax"
]
colors = [
    adjust_lightness('b', 0.3), adjust_lightness('tomato', 0.4),
    adjust_lightness('b', 0.8), adjust_lightness('tomato', 0.6),
    adjust_lightness('b', 1.4), adjust_lightness('tomato', 1.0),
    adjust_lightness('b', 1.6), adjust_lightness('tomato', 1.3),
    adjust_lightness('yellowgreen', 1.4), adjust_lightness('yellowgreen', 0.7),
    adjust_lightness('blueviolet', 1.3), "black"
]
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

'''
# get train, val and test datasets without frogs
xtrain, ytrain, xval, yval, xtest, ytest, _ = get_train_and_test_data(DATA, validation_test_split=True)
train_images, train_labels, unknown_train = split_up_some_classes(xtrain, ytrain)
val_images, val_labels, unknown_val = split_up_some_classes(xval, yval)
test_images, test_labels, unknown_test = split_up_some_classes(xtest, ytest)
ood = unknown_train + unknown_val + unknown_test


if DATA == "cars196":
    ood = ood[:8000]
elif DATA == "mnist":
    ood = ood[:34880]
ood = tf.reshape(ood, shape)

plt.figure(figsize=(10, 10))
for i in range(9):
    img = ood[i]
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.axis("off")
plt.show()

try:
    model = tf.keras.models.load_model(model_path)

except:
    print("no model found for ood detection")
    model = CNN(shape=shape[1:], classes=num_used_classes) if DATA != "cars196" else build_effnet(num_classes=num_used_classes)
    if DATA == "cifar10":
        model.load_weights("../models/classification/CNN_cifar100/cp.ckpt")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, restore_best_weights=True)
    if DATA == "cars196":
        early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=15, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)
    model.fit(train_images, train_labels, batch_size=128, shuffle=True, epochs=1000,
              validation_data=(val_images, val_labels), callbacks=[early_stop, rlrop])
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

    MCEstimator = MCDropoutEstimator(model, ood, num_used_classes, T=50, xval=val_images, yval=val_labels)
    DAEstimator = DataAugmentationEns(ood, num_used_classes, X_train=train_images, y_train=train_labels,
                                      X_val=val_images, y_val=val_labels, val=True, model_name=model_name)
    RISEstimator = RandomInitShuffleEns(ood, num_used_classes, X_train=train_images, y_train=train_labels,
                                        X_val=val_images, y_val=val_labels, val=True, model_name=model_name)
    BaEstimator = BaggingEns(ood, num_used_classes, X_train=train_images, y_train=train_labels,
                             X_val=val_images, y_val=val_labels, val=True, model_name=model_name)
    nuc_train = NeighborhoodUncertaintyClassifier(model, train_images, train_labels, val_images, val_labels, ood)
    nuc_val = NeighborhoodUncertaintyClassifier(model, xtrain_nuc_val, ytrain_nuc_val, xval_nuc_val, yval_nuc_val, ood)

    max_soft = tf.reduce_max(model.predict(ood, verbose=0), axis=-1).numpy()

    soft_ent_uncert_ood = tfd.Categorical(probs=model.predict(ood, verbose=0)).entropy().numpy()
    soft_ent_uncert_val = tfd.Categorical(probs=model.predict(val_images, verbose=0)).entropy().numpy()
    softmax_entropy = get_normalized_certainties(model.predict(val_images, verbose=0), val_labels,
                                                 soft_ent_uncert_val, soft_ent_uncert_ood)
    mcdr_se = MCEstimator.normalized_certainties_shannon_entropy()
    mcdr_mi = MCEstimator.normalized_certainties_mutual_information()
    bag_se = BaEstimator.normalized_certainties_shannon_entropy()
    bag_mi = BaEstimator.normalized_certainties_mutual_information()
    rand_se = RISEstimator.normalized_certainties_shannon_entropy()
    rand_mi = RISEstimator.normalized_certainties_mutual_information()
    aug_se = DAEstimator.normalized_certainties_shannon_entropy()
    aug_mi = DAEstimator.normalized_certainties_mutual_information()

    certainties = [mcdr_se, mcdr_mi,
                   bag_se, bag_mi, rand_se, rand_mi, aug_se, aug_mi,
                   nuc_train.certainties, nuc_val.certainties, 
                   softmax_entropy, max_soft]

    plt.xlabel("Schwellenwerte für die Konfidenz")
    plt.ylabel("Recall")
    plt.figure(figsize=(15, 15))

    for certs, method in zip(certainties, methods):
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
            json.dump(data, json_file, indent=4)
'''

with open('../Results/recalls_ood.json') as json_file:
    data = json.load(json_file)

plt.figure(figsize=(10, 2.8))
for i, d in enumerate(data):
    if i == 0:
        continue
    i -= 1
    plt.subplot(1, 3, i+1)

    for m, c in zip(methods, colors):
        if i == 0:
            plt.ylim(0.66, 1.01)
            if m == "NUC Train.":
                continue
        #if i == 2 or i == 1:
        #    if m == "ZIS SE" or m == "ZIS MI" or m == "Data Aug. SE" or m == "Data Aug. MI":
         #       continue
        mean = tf.reduce_mean(data[d][m], axis=-1)
        plt.plot(thresholds[2:], mean[2:], label=m, color=c, linestyle="--" if m == "Max Softmax" else "-",
                 linewidth=1., zorder=0 if m != "MC Drop. SE" and m != "MC Drop. MI" else 1)

    plt.xlabel("Certainty-Schwellenwert")
    plt.ylabel("Recall")
    plt.xticks([0.6, 0.7, 0.8, 0.9, 1.0])

    if i == 2:
        plt.title("CNN Cifar10")
    elif i == 1:
        plt.title("CNN Cifar100")
    else:
        plt.title("EfficientNet-B3 Cars")

plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
plt.subplots_adjust(left=0.07, right=0.83, bottom=0.16, top=0.9, wspace=0.32, hspace=0.35)
plt.savefig("../plots/ood.pdf")
plt.show()


'''plt.figure(figsize=(10, 2.8))
for i, d in enumerate(data):
    plt.subplot(1, 3, i+1)

    for m, c in zip(methods, colors):
        if i == 2 or i == 1:
            if m == "MC Drop. SE" or m == "MC Drop. MI" or m == "Softmax SE" or m == "NUC Train." or m == "NUC Valid.":
                continue
        mean = tf.reduce_mean(data[d][m], axis=-1)
        plt.plot(thresholds[2:], mean[2:], label=m, color=c, zorder=1 if re.match("B.*", m) else 0)

    plt.xlabel("Certainty-Schwellenwert")
    plt.ylabel("Recall")
    plt.xticks([0.6, 0.7, 0.8, 0.9, 1.0])

    if i == 2:
        plt.title("CNN Cifar10")
    elif i == 1:
        plt.title("CNN Cifar100")
    else:
        plt.title("EfficientNet-B3 Cars")

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.subplots_adjust(left=0.08, right=0.84, bottom=0.16, top=0.9, wspace=0.32, hspace=0.35)
plt.savefig("../plots/ood_ensembles.pdf")
plt.show()'''

