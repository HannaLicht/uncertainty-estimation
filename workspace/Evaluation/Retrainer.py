import random
import json
import sys
sys.path.append("/home/urz/hlichten")
print(sys.path)
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from functions import create_simple_model, get_train_and_test_data, ResNet
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.Ensemble import BaggingEns, DataAugmentationEns
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
import tensorflow_probability as tfp
tfd = tfp.distributions

STARTDATA = 1000
NUM_IMAGES = 1000
RUNS = 1
#PATH_TO_PRETRAINED_RESNET_10 = "../models/classification/simple_seq_model_fashion_mnist/cp.ckpt"
PATH_TO_PRETRAINED_RESNET_10 = "models/classification/ResNet_cifar10_" + str(STARTDATA) + "/cp.ckpt"
#PATH_TO_PRETRAINED_RESNET_100 = "../models/classification/simple_seq_model_mnist/cp.ckpt"
PATH_TO_PRETRAINED_RESNET_100 = "models/classification/ResNet_cifar100/cp.ckpt"


times_images_added = 9 if STARTDATA == 1000 else 5

(xleft, yleft), (xtest, ytest) = tf.keras.datasets.cifar10.load_data()
xleft = xleft.reshape(-1, 32, 32, 3) / 255.0
xtest = xtest.reshape(-1, 32, 32, 3) / 255.0

xtrain, ytrain = xleft[:STARTDATA], yleft[:STARTDATA]
xleft, yleft = xleft[STARTDATA:], yleft[STARTDATA:]

# comment if model already trained on STARTDATA of cifar10 training data
#train_base_model(PATH_TO_PRETRAINED_RESNET_100, PATH_TO_PRETRAINED_RESNET_10,
 #                xtrain, tf.keras.utils.to_categorical(ytrain.reshape((-1)), 10),
  #               xtest, tf.keras.utils.to_categorical(ytest.reshape((-1)), 10))


def mean(l):
    return sum(l)/len(l)


def train_base_model(checkpoint_path_old, checkpoint_path_new, X_train, y_train, X_test, y_test):
    """
    :param checkpoint_path_old: path of pretrained model cifar100
    :param checkpoint_path_new: path to save model further trained on the 'amount_of_data' first train data of
                                cifar10
    :return:
    """
    model = ResNet(classes=10)
    model.load_weights(checkpoint_path_old)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15,
                               restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)
    X_train, y_train = tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_new,
                                                     save_weights_only=True,
                                                     verbose=1)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              callbacks=[early_stop, cp_callback, rlrop], verbose=1, epochs=1000)


def prepare_model(path=PATH_TO_PRETRAINED_RESNET_10):
    model = ResNet(classes=10)
    # model = create_simple_model()
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
        '''
        if c2 is not None:
            index2 = sorted(range(len(cert)), key=cert.__getitem__, reverse=False)
            i, used_indices = 0, []
            while len(used_indices) < num_data:
                if index[i] not in used_indices:
                    self.xtrain.append(self.X_left[index[i]])
                    self.ytrain.append(self.y_left[index[i]])
                    used_indices.append(index[i])
                    if len(used_indices) == num_data:
                        break
                if index2[i] not in used_indices:
                    self.xtrain.append(self.X_left[index2[i]])
                    self.ytrain.append(self.y_left[index2[i]])
                    used_indices.append(index2[i])
                i += 1
        else:
        '''
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
                  callbacks=[early_stop, rlrop], verbose=1, epochs=1000, batch_size=128)
        loss, acc = model.evaluate(self.X_test, self.y_test, verbose=2)
        return acc, model


def retrain_with_ensemble(ensemble, metric):
    for _ in range(RUNS):
        model = prepare_model()
        retrainer = RetrainingEvaluator()

        with open('results_retrain.json') as json_file:
            data = json.load(json_file)

        for i in range(times_images_added):
            uncertainty_estimator = ensemble(retrainer.X_train, retrainer.y_train, retrainer.X_left, num_classes=10,
                                             model_name="ResNet_cifar10",
                                             X_test=retrainer.X_test, y_test=retrainer.y_test)

            if metric == "SE":
                certainties = uncertainty_estimator.get_certainties_by_SE()
                acc, model = retrainer.retrain(model, NUM_IMAGES, certainties)
                data[str(STARTDATA)][str(STARTDATA + (i+1)*NUM_IMAGES)]["Ensembles"][ensemble.__name__]["SE"] = \
                    data[str(STARTDATA)][str(STARTDATA+(i+1)*NUM_IMAGES)]["Ensembles"][ensemble.__name__]["SE"] + [acc]

            if metric == "MI":
                certainties = uncertainty_estimator.get_certainties_by_mutual_inf()
                acc, model = retrainer.retrain(model, NUM_IMAGES, certainties)
                data[str(STARTDATA)][str(STARTDATA + (i+1) * NUM_IMAGES)]["Ensembles"][ensemble.__name__]["MI"] = \
                    data[str(STARTDATA)][str(STARTDATA+(i+1)*NUM_IMAGES)]["Ensembles"][ensemble.__name__]["MI"] + [acc]

        with open('results_retrain.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)


def retrain_with_MCdrop(metric):
    for _ in range(RUNS):
        model = prepare_model()
        retrainer = RetrainingEvaluator()
        with open('results_retrain.json') as json_file:
            data = json.load(json_file)

        for i in range(times_images_added):
            clone_model = prepare_model()
            clone_model.set_weights(model.get_weights())
            uncertainty_estimator = MCDropoutEstimator(clone_model, retrainer.X_left, 10, 50)

            if metric == "SE":
                certainties = uncertainty_estimator.get_certainties_by_SE()
                acc, model = retrainer.retrain(model, NUM_IMAGES, certainties)
                data[str(STARTDATA)][str(STARTDATA + (i + 1) * NUM_IMAGES)]["MC_drop"]["SE"] = \
                    data[str(STARTDATA)][str(STARTDATA + (i + 1) * NUM_IMAGES)]["MC_drop"]["SE"] + [acc]

            if metric == "MI":
                certainties = uncertainty_estimator.get_certainties_by_mutual_inf()
                acc, model = retrainer.retrain(model, NUM_IMAGES, certainties)
                data[str(STARTDATA)][str(STARTDATA + (i + 1) * NUM_IMAGES)]["MC_drop"]["MI"] = \
                    data[str(STARTDATA)][str(STARTDATA + (i + 1) * NUM_IMAGES)]["MC_drop"]["MI"] + [acc]

        with open('results_retrain.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)


def retrain_with_nuc():
    for _ in range(RUNS):
        model = prepare_model()
        retrainer = RetrainingEvaluator()

        for i in range(times_images_added):
            uncertainty_estimator = \
                NeighborhoodUncertaintyClassifier(model, retrainer.X_train, tf.argmax(retrainer.y_train, axis=-1),
                                                  retrainer.X_test, tf.argmax(retrainer.y_test, axis=-1),
                                                  retrainer.X_left)
            acc, model = retrainer.retrain(model, NUM_IMAGES, uncertainty_estimator.certainties)

            with open('results_retrain.json') as json_file:
                data = json.load(json_file)
                data[str(STARTDATA)][str(STARTDATA + (i+1)*NUM_IMAGES)]["NUC"] = \
                    data[str(STARTDATA)][str(STARTDATA + (i+1)*NUM_IMAGES)]["NUC"] + [acc]
            with open('results_retrain.json', 'w') as json_file:
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

            with open('results_retrain.json') as json_file:
                data = json.load(json_file)
                data[str(STARTDATA)][str(STARTDATA + (i+1)*NUM_IMAGES)]["softmax_entropy"] = \
                    data[str(STARTDATA)][str(STARTDATA + (i+1)*NUM_IMAGES)]["softmax_entropy"] + [acc]
            with open('results_retrain.json', 'w') as json_file:
                json.dump(data, json_file, indent=4)


def retrain_with_random_data():
    for _ in range(RUNS):
        model = prepare_model()
        retrainer = RetrainingEvaluator()

        for i in range(times_images_added):
            certainties = [random.random() for _ in retrainer.y_left]
            acc, model = retrainer.retrain(model, NUM_IMAGES, certainties)

            with open('results_retrain.json') as json_file:
                data = json.load(json_file)
                data[str(STARTDATA)][str(STARTDATA + (i + 1) * NUM_IMAGES)]["random"] = \
                    data[str(STARTDATA)][str(STARTDATA + (i + 1) * NUM_IMAGES)]["random"] + [acc]
            with open('results_retrain.json', 'w') as json_file:
                json.dump(data, json_file, indent=4)


prepare_model().evaluate(xtest, tf.keras.utils.to_categorical(ytest.reshape((-1)), 10))
# check whether the classes are balanced in train dataset
print([list(ytrain).count(i) for i in range(10)])

#retrain_with_nuc()
#retrain_with_softmax_entropy()
#retrain_with_random_data()
#retrain_with_MCdrop("SE")
#retrain_with_MCdrop("MI")
retrain_with_ensemble(BaggingEns, "SE")
#retrain_with_ensemble(BaggingEns, "MI")
#retrain_with_ensemble(DataAugmentationEns, "SE")
#retrain_with_ensemble(DataAugmentationEns, "MI")

with open('results_retrain.json') as json_file:
    data = json.load(json_file)
    data = data[str(STARTDATA)]
    #data = data["simple_seq_model"]

numbers = [STARTDATA + i*NUM_IMAGES for i in range(times_images_added+1)]
rand = [mean(data[str(imgs)]["random"]) for imgs in numbers]
softmax = [mean(data[str(imgs)]["softmax_entropy"]) for imgs in numbers]
mc_se = [mean(data[str(imgs)]["MC_drop"]["SE"]) for imgs in numbers]
mc_mi = [mean(data[str(imgs)]["MC_drop"]["MI"]) for imgs in numbers]
bag_se = [mean(data[str(imgs)]["Ensembles"]["BaggingEns"]["SE"]) for imgs in numbers]
bag_mi = [mean(data[str(imgs)]["Ensembles"]["BaggingEns"]["MI"]) for imgs in numbers]
#aug_se = [mean(data[str(imgs)]["Ensembles"]["DataAugmentationEns"]["SE"]) for imgs in numbers]
#aug_mi = [mean(data[str(imgs)]["Ensembles"]["DataAugmentationEns"]["MI"]) for imgs in numbers]
nuc = [mean(data[str(imgs)]["NUC"]) for imgs in numbers]


methods_to_show = [rand,
                   softmax,
                   #mc_se,
                   #mc_mi,
                   #bag_se,
                   #bag_mi,
                   #aug_se,
                   #aug_mi,
                   nuc]
labels = ["random",
          "softmax entropy",
          #"MCdr SE",
          #"MCdr MI",
          #"bag SE",
          #"bag MI",
          #"aug SE"
          #"aug MI",
          "NUC"]

plt.figure(figsize=(14, 10))
for method, lbl in zip(methods_to_show, labels):
    plt.plot(numbers, method, label=lbl)
#plt.xticks([i for i in range(len(IMAGES))], IMAGES)
plt.xlabel("images to label")
plt.ylabel("Validation Accuracy")

plt.legend(loc="lower right")
plt.show()