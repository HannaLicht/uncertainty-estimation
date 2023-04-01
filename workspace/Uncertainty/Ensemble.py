import random
import re

import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from functions import CNN, build_effnet
from abc import abstractmethod
import tensorflow_probability as tfp
from Uncertainty.MC_Dropout import SamplingBasedEstimator

ENSEMBLE_LOCATION = "../Models/classification/ensembles"

"""
Ensemble members of these ensembles have same model architecture but are trained on different data samples
"""


class Ensemble(SamplingBasedEstimator):

    members = []

    def __init__(self, X, num_classes, path_to_ensemble="", X_train=None, y_train=None, X_val=None, y_val=None,
                 build_model_function=None, num_members=5, val=False):

        self.X, self.num_classes = X, num_classes
        if val:
            assert X_val is not None and y_val is not None
            self.xval, self.yval = X_val, y_val

        try:
            self.members = [tf.keras.models.load_model(path_to_ensemble + "/member_" + str(i)) for i in
                            range(num_members)]
            self.predict()

        except:
            assert X_train is not None and y_train is not None and X_val is not None and y_val is not None \
                   and build_model_function is not None
            print("Ensembe could not be found at path: " + str(path_to_ensemble))
            print("Ensemble will be trained now")
            self.init_new_ensemble(path_to_ensemble, X_train, y_train, X_val, y_val, build_model_function, num_members)

    @abstractmethod
    def prepare_data(self, X_train, y_train, num_members):
        """create ensemble based on certain approaches"""

    def init_new_ensemble(self, path_to_ensemble, X_train, y_train, X_val, y_val, build_model_function, num_members):
        train_imgs, train_lbls = self.prepare_data(X_train, y_train, num_members)
        self.init_members(build_model_function, num_members)

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
        rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)

        if build_model_function != CNN:
            # transferlearning
            # first step
            for index, (model, imgs, lbls) in enumerate(zip(self.members, train_imgs, train_lbls)):
                model.fit(imgs, lbls, validation_data=(X_val, y_val), epochs=1000,
                          batch_size=128 if len(lbls) >= 1000 else 32, callbacks=[early_stop_transfer])
                # if convergence: begin second step
                model.trainable = True
                optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3 if build_model_function != build_effnet else 1e-4)
                model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        # train ensemble members
        for index, (model, imgs, lbls) in enumerate(zip(self.members, train_imgs, train_lbls)):
            model.fit(imgs, lbls, validation_data=(X_val, y_val), epochs=1000,
                      batch_size=128 if len(lbls) >= 1000 else 32, callbacks=[early_stop, rlrop])
            if path_to_ensemble != "":
                model.save(path_to_ensemble + "/member_" + str(index))

        self.predict()

    def init_members(self, build_model_function, num_members):
        self.members = [build_model_function(self.num_classes) for _ in range(num_members)]
        for member in self.members:
            optimizer = tf.keras.optimizers.Adam(lr=1e-2)
            member.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def predict(self):
        self.predictions = [model.predict(self.X, batch_size=32) for model in self.members]
        self.p_ens = tf.math.reduce_mean(self.predictions, axis=0)
        if self.xval is not None:
            self.val_predictions = [model.predict(self.xval, batch_size=32) for model in self.members]
            self.val_p_ens = tf.math.reduce_mean(self.val_predictions, axis=0)


class BaggingEns(Ensemble):

    def __init__(self, X, num_classes, path_to_ensemble="", X_train=None, y_train=None, X_val=None, y_val=None,
                 build_model_function=None, num_members=5, val=False):
        self.estimator_name = "Ensemble - Bagging"
        super().__init__(X, num_classes, path_to_ensemble, X_train, y_train, X_val, y_val, build_model_function,
                        num_members, val)

    def prepare_data(self, xtrain, ytrain, num_members):
        train_imgs, train_lbls = [], []
        for i in range(num_members):
            rand = tf.random.uniform(shape=[len(xtrain)], minval=0, maxval=len(xtrain) - 1, dtype=tf.dtypes.int64)
            train_imgs.append(tf.gather(xtrain, rand))
            train_lbls.append(tf.gather(ytrain, rand))

        return train_imgs, train_lbls


class RandomInitShuffleEns(Ensemble):

    def __init__(self, X, num_classes, path_to_ensemble="", X_train=None, y_train=None, X_val=None, y_val=None,
                 build_model_function=None, num_members=5, val=False):
        self.estimator_name = "Ensemble - Random Initialization & Data Shuffle"
        super().__init__(X, num_classes, path_to_ensemble, X_train, y_train, X_val, y_val, build_model_function,
                         num_members, val)

    def prepare_data(self, xtrain, ytrain, num_members):
        return [xtrain for _ in range(num_members)], [ytrain for _ in range(num_members)]


class DataAugmentationEns(Ensemble):

    def __init__(self, X, num_classes, path_to_ensemble="", X_train=None, y_train=None, X_val=None, y_val=None,
                 build_model_function=None, num_members=5, val=False):
        self.estimator_name = "Ensemble - Data Augmentation"
        super().__init__(X, num_classes, path_to_ensemble, X_train, y_train, X_val, y_val, build_model_function,
                         num_members, val)

    def init_new_ensemble(self, path_to_ensemble, X_train, y_train, X_val, y_val, build_model_function, num_members):
        if y_train[0].shape == (128, 128, 1):
            X_train = tf.reshape(X_train, (-1, 128, 128, 3))
            X_val = tf.reshape(X_val, (-1, 128, 128, 3))
            y_train = tf.reshape(y_train, (-1, 128, 128, 1))
            y_val = tf.reshape(y_val, (-1, 128, 128, 1))

        data_generator = self.prepare_data(X_train, y_train, num_members)
        self.init_members(build_model_function, num_members)

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
        rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)

        if build_model_function != CNN:
            # transferlearning
            # first step
            for index, model in enumerate(self.members):
                model.fit(data_generator, validation_data=(X_val, y_val), epochs=1000, callbacks=[early_stop_transfer])
                # if convergence: begin second step
                model.trainable = True
                optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3 if build_model_function != build_effnet else 1e-4)
                model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        # train ensemble members
        for index, model in enumerate(self.members):
            model.fit(data_generator, validation_data=(X_val, y_val), epochs=1000, callbacks=[early_stop, rlrop])
            if path_to_ensemble != "":
                model.save(path_to_ensemble + "/member_" + str(index))

        self.predict()

    def prepare_data(self, xtrain, ytrain, num_members):

        data_augmentation = ImageDataGenerator(rotation_range=1, width_shift_range=0.05, height_shift_range=0.05,
                                               zoom_range=.1, horizontal_flip=True, fill_mode='reflect')
        data_augmentation.fit(xtrain, augment=True)
        iterator = data_augmentation.flow(xtrain, ytrain, shuffle=True, batch_size=128 if len(ytrain) >= 1000 else 32)

        # uncomment for a plot of different augmentations of an image
        '''plt.figure(figsize=(10, 10))
        for i in range(9):
            iter = data_augmentation.flow(xtrain, ytrain, shuffle=False, batch_size=128)
            augmented_batch = next(iter)
            plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_batch[0][2])
            plt.axis("off")
        plt.show()'''

        return iterator
