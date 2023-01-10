import os
import re
import tensorflow as tf
from keras.applications import efficientnet
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
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

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
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


# preprocess function
def resize_with_crop_effnet(image, label):
    i = image
    i = tf.cast(i, tf.float32)
    i = tf.image.resize_with_crop_or_pad(i, 300, 300)
    i = efficientnet.preprocess_input(i)
    return i, label


def get_train_and_test_data(data, validation_test_split=False):
    if data == "mnist":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
        X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
        classes = 10

    elif data == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
        X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
        classes = 10

    elif data == "cifar100":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()
        X_train = X_train.reshape(-1, 32, 32, 3) / 255.0
        X_test = X_test.reshape(-1, 32, 32, 3) / 255.0
        y_train = tf.keras.utils.to_categorical(y_train.reshape((-1)), 100)
        y_test = tf.keras.utils.to_categorical(y_test.reshape((-1)), 100)
        classes = 100

    elif data == "cifar10":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        X_train = X_train.reshape(-1, 32, 32, 3) / 255.0
        X_test = X_test.reshape(-1, 32, 32, 3) / 255.0
        y_train = tf.keras.utils.to_categorical(y_train.reshape((-1)), 10)
        y_test = tf.keras.utils.to_categorical(y_test.reshape((-1)), 10)
        classes = 10

    elif data == "pets":
        # oxford-IIIT Pets dataset
        dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

        def normalize(input_image, input_mask):
            input_image = tf.cast(input_image, tf.float32) / 255.0
            input_mask -= 1  # change labels from (1, 2, 3) to (0, 1, 2)
            return input_image, input_mask

        def load_image(datapoint):
            input_image = tf.image.resize(datapoint['image'], (128, 128))
            input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
            input_image, input_mask = normalize(input_image, input_mask)
            return input_image, input_mask

        train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        X_train, y_train = list(zip(*train_images))
        test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        X_test, y_test = list(zip(*test_images))
        classes = 3

    elif data == "imagenet":
        # https://medium.com/analytics-vidhya/how-to-train-a-neural-network-classifier-on-imagenet-using-tensorflow-2-ede0ea3a35ff
        # Get imagenet labels
        labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                              'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
        imagenet_labels = list(open(labels_path).read().splitlines())

        # Set data_dir to a read-only storage of .tar files
        # Set write_dir to a w/r storage
        data_dir = '../ImageNet-ILSVRC2012/tars'
        write_dir = '../ImageNet-ILSVRC2012/data'

        # Construct a tf.data.Dataset
        download_config = tfds.download.DownloadConfig(
            extract_dir=os.path.join(write_dir, 'extracted'),
            manual_dir=data_dir
        )
        download_and_prepare_kwargs = {
            'download_dir': os.path.join(write_dir, 'downloaded'),
            'download_config': download_config,
        }
        ds_train, ds_val = tfds.load('imagenet2012_subset/10pct',
                                     data_dir=os.path.join(write_dir, 'data'),
                                     split=['train', 'validation'],
                                     shuffle_files=False,
                                     download=True,
                                     as_supervised=True,
                                     download_and_prepare_kwargs=download_and_prepare_kwargs
                                     )
        ds_test = tfds.load('imagenet_a',
                            data_dir=os.path.join(write_dir, 'data'),
                            split="test",
                            shuffle_files=False,
                            download=True,
                            as_supervised=True,
                            download_and_prepare_kwargs=download_and_prepare_kwargs
                            )

        train_images = ds_train.map(resize_with_crop_effnet)
        val_images = ds_val.map(resize_with_crop_effnet)
        test_images = ds_test.map(resize_with_crop_effnet)

        X_train, y_train = list(zip(*train_images))
        X_val, y_val = list(zip(*val_images))
        X_test, y_test = list(zip(*test_images))
        X_val, y_val = tf.reshape(X_val, (-1, 300, 300, 3)), tf.keras.utils.to_categorical(y_val, 1000)
        X_test, y_test = tf.reshape(X_test, (-1, 300, 300, 3)), tf.keras.utils.to_categorical(y_test, 1000)

        if validation_test_split:
            return X_train, y_train, X_val, y_val, X_test, y_test, 1000
        else:
            return X_train, y_train, X_test, y_test, 1000

    else:
        raise NotImplementedError

    if validation_test_split:
        X_val, y_val = X_train[int((4./5.)*len(X_train)):], y_train[int((4./5.)*len(y_train)):]
        X_train, y_train = X_train[:int((4./5.)*len(X_train))], y_train[:int((4./5.)*len(y_train))]
        return X_train, y_train, X_val, y_val, X_test, y_test, classes

    else:
        return X_train, y_train, X_test, y_test, classes

'''
def get_test_data(data):
    """
    :param data: name of dataset
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
'''
