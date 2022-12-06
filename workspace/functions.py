import re
import tensorflow as tf
from keras.applications import efficientnet
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from matplotlib import pyplot as plt

tfd = tfp.distributions

translate = {
    0: 0,
    1: 217,
    2: 482,
    3: 491,
    4: 497,
    5: 566,
    6: 569,
    7: 571,
    8: 574,
    9: 701
}


def imgnette_to_imgnet_lbls(y):
    y_ = []
    for lbl in y:
        y_.append(translate.get(lbl.numpy()))
    return y_


def create_simple_model():
    inp = tf.keras.Input((28, 28))
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dense(512, activation='relu', input_shape=(784,))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model


def CNN(shape=(32, 32, 3), classes=100):
    x_input = tf.keras.layers.Input(shape)
    x = tf.keras.layers.ZeroPadding2D(3)(x_input)
    # initial conv layer
    x = tf.keras.layers.Conv2D(32, 7, padding='same', strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x_skip = x
    x = tf.keras.layers.Conv2D(32, 3, padding='same', strides=2)(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x_skip = tf.keras.layers.Conv2D(32, 1,  strides=2)(x_skip)
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)

    x_skip = x
    x = tf.keras.layers.Conv2D(64, 3, padding='same', strides=2)(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x_skip = tf.keras.layers.Conv2D(64, 1,  strides=2)(x_skip)
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)

    # begin of head: always trainable
    x = tf.keras.layers.AveragePooling2D(2, padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(100, activation='softmax' if classes == 100 else 'relu')(x)
    if classes == 10:
      x = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=x_input, outputs=x)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_dropout_rate(model):
    layers = model.get_config().get('layers')
    dropout = []

    for layer in layers:
        if layer.get('class_name') == 'Dropout':
            dropout.append(layer.get('config').get('rate'))
    dropout = tf.math.reduce_mean(dropout)

    return dropout


def get_train_and_test_data(data):
    if data == "mnist":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
        X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
        return X_train, y_train, X_test, y_test, 10

    elif data == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
        X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
        return X_train, y_train, X_test, y_test, 10

    elif data == "cifar100":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()
        X_train = X_train.reshape(-1, 32, 32, 3) / 255.0
        X_test = X_test.reshape(-1, 32, 32, 3) / 255.0
        y_train = tf.keras.utils.to_categorical(y_train.reshape((-1)), 100)
        y_test = tf.keras.utils.to_categorical(y_test.reshape((-1)), 100)
        return X_train, y_train, X_test, y_test, 100

    elif data == "cifar10":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        X_train = X_train.reshape(-1, 32, 32, 3) / 255.0
        X_test = X_test.reshape(-1, 32, 32, 3) / 255.0
        y_train = tf.keras.utils.to_categorical(y_train.reshape((-1)), 10)
        y_test = tf.keras.utils.to_categorical(y_test.reshape((-1)), 10)
        return X_train, y_train, X_test, y_test, 10


def get_train_data(data):
    """
    :param data:
    :return: train images, train labels, number of classes
    """

    if data == "mnist":
        (X_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
        return X_train, y_train, 10
    elif data == "fashion_mnist":
        (X_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
        return X_train, y_train, 10
    elif data == "cifar10":
        (X_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
        X_train = X_train.reshape(-1, 32, 32, 3) / 255.0
        return X_train, y_train.reshape((-1)), 10
    else:
        print("dataset not available")
        return None


def get_test_data(data):
    """
    :param data:
    :param num_data:
    :return: test images, test labels, number of classes
    """
    if data == "mnist":
        _, (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
        return X_test, y_test, 10

    elif data == "fashion_mnist":
        _, (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
        return X_test, y_test, 10

    elif data == "cifar10":
        _, (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        X_test = X_test.reshape(-1, 32, 32, 3) / 255.0
        y_test = tf.keras.utils.to_categorical(y_test.reshape((-1)), 10)
        return X_test, y_test, 10

    elif data == "cifar100":
        _, (X_test, y_test) = tf.keras.datasets.cifar100.load_data()
        X_test = X_test.reshape(-1, 32, 32, 3) / 255.0
        y_test = tf.keras.utils.to_categorical(y_test.reshape((-1)), 100)
        return X_test, y_test, 100

    elif data == "imagenette":
        dir = '../datasets/imagenette'
        #testset = tfds.load('imagenette', split='validation', data_dir=dir, as_supervised=True, batch_size=num_data)
        testset = tfds.load('imagenette', split='validation', data_dir=dir, as_supervised=True, batch_size=10)
        testset = testset.map(resize_with_crop_effnet)
        for img, lbl in testset:
            X_test = img
            y_test = lbl
            break
        y_test = imgnette_to_imgnet_lbls(y_test)
        y_test = tf.keras.utils.to_categorical(y_test, 1000)  # model output 1000 classes (imagenet)
        return X_test, y_test, 1000

    else:
        print("dataset not available")
        return None


# preprocess functions:
def resize_with_crop_effnet(image, label):
    i = image
    i = tf.cast(i, tf.float32)
    i = tf.image.resize_with_crop_or_pad(i, 300, 300)
    i = efficientnet.preprocess_input(i)
    return i, label
