import random
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from functions import create_simple_model, CNN
from abc import abstractmethod
import tensorflow_probability as tfp

from models.semantic_segmentation.train_and_save_modified_UNet import unet_model
from uncertainty.MC_Dropout import SamplingBasedEstimator

ENSEMBLE_LOCATION = "../models/classification/ensembles"

"""
Ensemble members have same model architecture but are trained on different data samples
"""


class Ensemble(SamplingBasedEstimator):

    members = []

    def __init__(self, X_train, y_train, X, num_classes, model_name, path_to_ensemble="", X_val=None, y_val=None,
                 optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], num_members=5, val=False):

        self.X, self.num_classes = X, num_classes
        if val:
            assert X_val is not None and y_val is not None
            self.xval, self.yval = X_val, y_val
        try:
            self.members = [tf.keras.models.load_model(path_to_ensemble + "/member_" + str(i)) for i in
                            range(num_members)]
            self.predict()
        except:
            print("Ensembe could not be found at path: " + str(path_to_ensemble))
            print("Ensemble will be trained now")
            self.init_new_ensemble(path_to_ensemble, X_train, y_train, X_val, y_val, model_name, num_members,
                                   optimizer, loss, metrics)

    @abstractmethod
    def prepare_data(self, X_train, y_train, num_members):
        """create ensemble based on certain approaches"""

    def init_new_ensemble(self, path_to_ensemble, X_train, y_train, X_val, y_val, model_name, num_members,
                          optimizer, loss, metrics):
        if y_train[0].shape == (128, 128, 1):
            X_train = tf.reshape(X_train, (-1, 128, 128, 3))
            X_val = tf.reshape(X_val, (-1, 128, 128, 3))
            y_train = tf.reshape(y_train, (-1, 128, 128, 1))
            y_val = tf.reshape(y_val, (-1, 128, 128, 1))

        train_imgs, train_lbls = self.prepare_data(X_train, y_train, num_members)

        if model_name == "simple_seq_model":
            self.members = [create_simple_model() for _ in range(num_members)]
        elif model_name == "CNN_cifar10":
            self.members = [CNN(classes=10) for _ in range(num_members)]
            for member in self.members:
                member.load_weights("../models/classification/CNN_cifar100/cp.ckpt")
                member.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        elif model_name == "CNN_cifar100":
            self.members = [CNN(classes=100) for _ in range(num_members)]
        elif model_name == "modified_UNet":
            self.members = [unet_model(output_channels=self.num_classes) for _ in range(num_members)]
        else:
            raise NotImplementedError

        rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)

        # train ensemble members
        for index, (model, imgs, lbls) in enumerate(zip(self.members, train_imgs, train_lbls)):
            model.fit(imgs, lbls, validation_data=(X_val, y_val), epochs=1000,
                      batch_size=128 if len(lbls) >= 1000 else 32, callbacks=[early_stop, rlrop])
            if path_to_ensemble is not "":
                model.save(path_to_ensemble + "/member_" + str(index))

        self.predict()

    def predict(self):
        self.predictions = [model.predict(self.X, batch_size=32) for model in self.members]
        self.p_ens = tf.math.reduce_mean(self.predictions, axis=0)
        if self.xval is not None:
            self.val_predictions = [model.predict(self.xval, batch_size=32) for model in self.members]
            self.val_p_ens = tf.math.reduce_mean(self.val_predictions, axis=0)


class BaggingEns(Ensemble):

    def __init__(self, X_train, y_train, X, num_classes, model_name, path_to_ensemble="", X_val=None, y_val=None,
                 optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], num_members=5, val=False):
        self.estimator_name = "Ensemble - Bagging"
        super().__init__(X_train, y_train, X, num_classes, model_name, path_to_ensemble, X_val, y_val,
                         optimizer, loss, metrics, num_members, val)

    def prepare_data(self, xtrain, ytrain, num_members):
        train_imgs, train_lbls = [], []
        for i in range(num_members):
            rand = tf.random.uniform(shape=[len(xtrain)], minval=0, maxval=len(xtrain) - 1, dtype=tf.dtypes.int64)
            train_imgs.append(tf.gather(xtrain, rand))
            train_lbls.append(tf.gather(ytrain, rand))

        return train_imgs, train_lbls


class RandomInitShuffleEns(Ensemble):

    def __init__(self, X_train, y_train, X, num_classes, model_name, path_to_ensemble="", X_val=None, y_val=None,
                 optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], num_members=5, val=False):
        self.estimator_name = "Ensemble - Random Initialization & Data Shuffle"
        super().__init__(X_train, y_train, X, num_classes, model_name, path_to_ensemble, X_val, y_val,
                         optimizer, loss, metrics, num_members, val)

    def prepare_data(self, xtrain, ytrain, num_members):
        return [xtrain for _ in range(num_members)], [ytrain for _ in range(num_members)]


class DataAugmentationEns(Ensemble):

    def __init__(self, X_train, y_train, X, num_classes, model_name, path_to_ensemble="", X_val=None, y_val=None,
                 optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], num_members=5, val=False):
        self.estimator_name = "Ensemble - Data Augmentation"
        super().__init__(X_train, y_train, X, num_classes, model_name, path_to_ensemble, X_val, y_val,
                         optimizer, loss, metrics, num_members, val)

    def init_new_ensemble(self, path_to_ensemble, X_train, y_train, X_val, y_val, model_name, num_members,
                          optimizer, loss, metrics):
        if y_train[0].shape == (128, 128, 1):
            X_train = tf.reshape(X_train, (-1, 128, 128, 3))
            X_val = tf.reshape(X_val, (-1, 128, 128, 3))
            y_train = tf.reshape(y_train, (-1, 128, 128, 1))
            y_val = tf.reshape(y_val, (-1, 128, 128, 1))

        data_generator = self.prepare_data(X_train, y_train, num_members)

        if model_name == "simple_seq_model":
            self.members = [create_simple_model() for _ in range(num_members)]
        elif model_name == "CNN_cifar10":
            self.members = [CNN(classes=10) for _ in range(num_members)]
            for member in self.members:
                member.load_weights("../models/classification/CNN_cifar100/cp.ckpt")
                member.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        elif model_name == "CNN_cifar100":
            self.members = [CNN(classes=100) for _ in range(num_members)]
        elif model_name == "modified_UNet":
            self.members = [unet_model(output_channels=self.num_classes) for _ in range(num_members)]
        else:
            raise NotImplementedError

        rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)

        # train ensemble members
        for index, model in enumerate(self.members):
            model.fit(data_generator, validation_data=(X_val, y_val), epochs=1000, callbacks=[early_stop, rlrop])
            if path_to_ensemble is not "":
                model.save(path_to_ensemble + "/member_" + str(index))

        self.predict()

    def prepare_data(self, xtrain, ytrain, num_members):
        if ytrain[0].shape == (128, 128, 1):        # semantic segmentation

            def add_noise(img):
                """Add random noise to an image"""
                variability = 0.01
                deviation = variability * random.random()
                noise = tfp.distributions.Normal(tf.fill(img.shape, 0.), tf.fill(img.shape, deviation))
                img += noise.sample()
                tf.clip_by_value(img, 0., 255.)
                return img

            data_augmentation = ImageDataGenerator(preprocessing_function=add_noise)

            '''
            mask_datagen = ImageDataGenerator(**data_gen_args)

            # Provide the same seed and keyword arguments to the fit and flow methods seed = 1
            image_datagen.fit(xtrain, augment=True, seed=[42, 35, 2, 7, 8])
            mask_datagen.fit(ytrain, augment=True, seed=[42, 35, 2, 7, 8])

            image_iterator = image_datagen.flow(xtrain, batch_size=128 if len(ytrain) >= 1000 else 32, seed=1)
            mask_iterator = mask_datagen.flow(ytrain, batch_size=128 if len(ytrain) >= 1000 else 32, seed=1)

            plt.figure(figsize=(8, 16))
            for i in range(4):
                image_iterator = image_datagen.flow(xtrain, batch_size=128 if len(ytrain) >= 1000 else 32, seed=1)
                mask_iterator = mask_datagen.flow(ytrain, batch_size=128 if len(ytrain) >= 1000 else 32, seed=1)
                augmented_image = next(image_iterator)
                augmented_mask = next(mask_iterator)
                plt.subplot(4, 2, 2 * i + 1)
                plt.imshow(augmented_image[0])
                plt.subplot(4, 2, 2 * i + 2)
                plt.imshow(augmented_mask[0])
                plt.axis("off")
            plt.show()

            train_iterator = zip(image_iterator, mask_iterator)
            return train_iterator
            '''
        else:
            data_augmentation = ImageDataGenerator(rotation_range=1, width_shift_range=0.05, height_shift_range=0.05,
                                                   zoom_range=.1, horizontal_flip=True, fill_mode='reflect')
        data_augmentation.fit(xtrain, augment=True)
        iterator = data_augmentation.flow(xtrain, ytrain, shuffle=True, batch_size=128 if len(ytrain) >= 1000 else 32)

        '''plt.figure(figsize=(10, 10))
        for i in range(9):
            iter = data_augmentation.flow(xtrain, ytrain, shuffle=False, batch_size=128)
            augmented_batch = next(iter)
            plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_batch[0][2])
            plt.axis("off")
        plt.show()'''

        return iterator
