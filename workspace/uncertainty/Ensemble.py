import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from functions import create_simple_model, ResNet
from abc import abstractmethod
from uncertainty.MC_Dropout import SamplingBasedEstimator

ENSEMBLE_LOCATION = "../models/classification/ensembles"

"""
Ensemble members have same model architecture but are trained on different data samples
"""


class Ensemble(SamplingBasedEstimator):

    members = []

    def __init__(self, X_train, y_train, X, num_classes=10, model_name=None, path_to_ensemble=None, X_test=None,
                 y_test=None, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], num_members=5):

        self.X, self.num_classes = X, num_classes
        try:
            self.members = [tf.keras.models.load_model(path_to_ensemble + "/member_" + str(i)) for i in
                            range(num_members)]
            self.predict()
        except:
            print("Ensembe could not be found at path: " + str(path_to_ensemble))
            print("Ensemble will be trained now")
            self.init_new_ensemble(path_to_ensemble, X_train, y_train, X_test, y_test, model_name, num_members,
                                   optimizer, loss, metrics)


    @abstractmethod
    def prepare_data(self, X_train, y_train, num_members):
        """create ensemble based on certain approaches"""

    def init_new_ensemble(self, path_to_ensemble, X_train, y_train, X_test, y_test, model_name, num_members,
                          optimizer, loss, metrics):
        train_imgs, train_lbls = self.prepare_data(X_train, y_train, num_members)

        if model_name == "simple_seq_model":
            self.members = [create_simple_model() for _ in range(num_members)]
        elif model_name == "ResNet_cifar10":
            self.members = [ResNet(classes=10) for _ in range(num_members)]
            for member in self.members:
                member.load_weights("../models/classification/ResNet_cifar100/cp.ckpt")
                member.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        elif model_name == "ResNet_cifar100":
            self.members = [ResNet(classes=100) for _ in range(num_members)]
        else:
            raise NotImplementedError

        rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)

        # train ensemble members
        for index, (model, imgs, lbls) in enumerate(zip(self.members, train_imgs, train_lbls)):
            model.fit(imgs, lbls, validation_data=(X_test, y_test), epochs=1000, batch_size=128, callbacks=[early_stop, rlrop])
            if path_to_ensemble is not None:
                model.save(path_to_ensemble + "/member_" + str(index))

        self.predict()

    def predict(self):
        self.predictions = [model.predict(self.X, batch_size=32) for model in self.members]
        self.p_ens = tf.math.reduce_mean(self.predictions, axis=0)


class BaggingEns(Ensemble):

    def __init__(self, X_train, y_train, X, num_classes, model_name=None, path_to_ensemble=None, X_test=None,
                 y_test=None, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], num_members=5):
        self.estimator_name = "Ensemble - Bagging"
        super().__init__(X_train, y_train, X, num_classes, model_name, path_to_ensemble, X_test, y_test,
                         optimizer, loss, metrics, num_members)

    def prepare_data(self, xtrain, ytrain, num_members):
        train_imgs, train_lbls = [], []
        for i in range(num_members):
            rand = tf.random.uniform(shape=[len(xtrain)], minval=0, maxval=len(xtrain) - 1, dtype=tf.dtypes.int64)
            train_imgs.append(tf.gather(xtrain, rand))
            train_lbls.append(tf.gather(ytrain, rand))

        return train_imgs, train_lbls


class DataAugmentationEns(Ensemble):

    def __init__(self, X_train, y_train, X, num_classes, model_name=None, path_to_ensemble=None, X_test=None,
                 y_test=None, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], num_members=5):
        self.estimator_name = "Ensemble - Data Augmentation"
        super().__init__(X_train, y_train, X, num_classes, model_name, path_to_ensemble, X_test, y_test,
                         optimizer, loss, metrics, num_members)

    def init_new_ensemble(self, path_to_ensemble, X_train, y_train, X_test, y_test, model_name, num_members,
                          optimizer, loss, metrics):
        data_generator = self.prepare_data(X_train, y_train, num_members)

        if model_name == "simple_seq_model":
            self.members = [create_simple_model() for _ in range(num_members)]
        elif model_name == "ResNet_cifar10":
            self.members = [ResNet(classes=10) for _ in range(num_members)]
            for member in self.members:
                member.load_weights("../models/classification/ResNet_cifar100/cp.ckpt")
                member.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        elif model_name == "ResNet_cifar100":
            self.members = [ResNet(classes=100) for _ in range(num_members)]
        else:
            raise NotImplementedError

        rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)

        # train ensemble members
        for index, model in enumerate(self.members):
            model.fit(data_generator, validation_data=(X_test, y_test), epochs=1000, callbacks=[early_stop, rlrop])
            if path_to_ensemble is None:
                model.save(ENSEMBLE_LOCATION + "/" + self.estimator_name + "/member_" + str(index))
            else:
                model.save(path_to_ensemble + "/member_" + str(index))

        self.predict()

    def prepare_data(self, xtrain, ytrain, num_members):
        data_augmentation = ImageDataGenerator(rotation_range=1, width_shift_range=0.05, height_shift_range=0.05,
                                               zoom_range=.1, horizontal_flip=True, fill_mode='reflect')
        data_augmentation.fit(xtrain, augment=True)
        iterator = data_augmentation.flow(xtrain, ytrain, shuffle=True, batch_size=128)
        '''
        plt.figure(figsize=(10, 10))
        for i in range(9):
            augmented_image = next(iterator)
            plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_image[0])
            plt.axis("off")
        plt.show()
        '''
        return iterator
