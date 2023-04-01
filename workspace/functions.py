import numpy as np
import tensorflow as tf
from keras.applications import efficientnet
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
tfd = tfp.distributions


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


COLORS = {
    "NUC Tr": "green",
    "NUC Va": adjust_lightness("yellowgreen", 0.85),
    "MCD PE": adjust_lightness('b', 0.4),
    "MCD MI": adjust_lightness('tomato', 0.4),
    "Bag PE": "b",
    "Bag MI": adjust_lightness('tomato', 0.7),
    "DA PE": adjust_lightness('b', 1.5),
    "DA MI": adjust_lightness('tomato', 1.1),
    "SE": "mediumturquoise",
}


def CNN(classes=100, shape=(32, 32, 3)):
    x_input = tf.keras.layers.Input(shape)
    # initial conv layer
    x = tf.keras.layers.Conv2D(32, 5, padding='same')(x_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = tf.keras.layers.Conv2D(32, 3, padding='same')(x)
    x = tf.keras.layers.MaxPool2D(strides=2, padding='same')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(x)
    x = tf.keras.layers.MaxPool2D(strides=2, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.AveragePooling2D(2, padding='same')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=x_input, outputs=x)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_effnet(num_classes, img_size=300):
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    model = tf.keras.applications.EfficientNetB3(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def CNN_transfer_learning(classes=10, path_pretrained_model="../models/classification/CNN_cifar100"):
    model_body = CNN(shape=(32, 32, 3), classes=100)
    model_body.load_weights(path_pretrained_model)
    out = model_body.layers[-2].output
    model = tf.keras.Model(inputs=model_body.input, outputs=out)

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(model.output)
    outputs = tf.keras.layers.Dense(classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(model_body.input, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# use it for the cars196 dataset: many classes but small dataset -> make sure a split evently divides dataset
def split_validation_from_train(xtrain, ytrain, num_classes, num_imgs_per_class):
    count = np.zeros(num_classes)
    xval, yval, x_train, y_train = [], [], [], []

    for i, (img, y) in enumerate(zip(xtrain, ytrain)):
        lbl = tf.argmax(y, axis=-1)
        if count[lbl] < num_imgs_per_class:
            count[lbl] = count[lbl] + 1
            xval.append(img)
            yval.append(y)
        else:
            x_train.append(img)
            y_train.append(y)

    xval, yval = tf.reshape(xval, (-1, 300, 300, 3)), tf.reshape(yval, (-1, num_classes))
    x_train, y_train = tf.reshape(x_train, (-1, 300, 300, 3)), tf.reshape(y_train, (-1, num_classes))

    return x_train, y_train, xval, yval


# preprocess function for EfficientNet
def resize_with_crop_effnet(image, label):
    i = image
    i = tf.cast(i, tf.float32)
    i = tf.image.resize_with_crop_or_pad(i, 300, 300)
    i = efficientnet.preprocess_input(i)
    return i, label


def get_data(data, num_data=None, active_learning=False):
    if data == "cifar100":
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
        # oxford-IIIT Pets dataset for the U-Net (semantic segmentation)
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

    elif data == "cars196":
        # https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
        IMG_SIZE = 300

        (ds_train, ds_test), ds_info = tfds.load(
            "cars196", split=["train", "test"], with_info=True, as_supervised=True, shuffle_files=False
        )
        NUM_CLASSES = ds_info.features["label"].num_classes

        size = (IMG_SIZE, IMG_SIZE)
        ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
        ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))

        # One-hot / categorical encoding
        def input_preprocess(image, label):
            label = tf.one_hot(label, NUM_CLASSES)
            return image, label

        ds_train = ds_train.map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.map(input_preprocess)

        xtrain, ytrain = list(zip(*ds_train))
        xtest, ytest = list(zip(*ds_test))

        # make sure the classes among valid. data are uniformly distributed (each class has 8 images -> 8*196 = 1568)
        xtrain, ytrain, xval, yval = split_validation_from_train(xtrain, ytrain, NUM_CLASSES, num_imgs_per_class=8)

        xtest, ytest = tf.reshape(xtest, (-1, IMG_SIZE, IMG_SIZE, 3)), tf.reshape(ytest, (-1, 196))

        return xtrain, ytrain, xval, yval, xtest, ytest, NUM_CLASSES

    else:
        raise NotImplementedError

    if num_data is not None:
        X_left, y_left = X_train, y_train
        X_val, y_val = X_train[:int(num_data/4)], y_train[:int(num_data/4)]
        X_train, y_train = X_train[int(num_data/4):int(5*num_data/4)], y_train[int(num_data/4):int(5*num_data/4)]
        if active_learning:
            X_val = tf.concat([X_val, X_left[-int(3*num_data/4):]], axis=0)
            y_val = tf.concat([y_val, y_left[-int(3*num_data/4):]], axis=0)
            X_left, y_left = X_left[int(5*num_data/4):-int(3*num_data/4)], y_left[int(5*num_data/4):-int(3*num_data/4)]
            return X_train, y_train, X_val, y_val, X_test, y_test, classes, X_left, y_left
        return X_train, y_train, X_val, y_val, X_test, y_test, classes
    else:
        X_val, y_val = X_train[int((4. / 5.) * len(X_train)):], y_train[int((4. / 5.) * len(y_train)):]
        X_train, y_train = X_train[:int((4. / 5.) * len(X_train))], y_train[:int((4. / 5.) * len(y_train))]
        return X_train, y_train, X_val, y_val, X_test, y_test, classes

