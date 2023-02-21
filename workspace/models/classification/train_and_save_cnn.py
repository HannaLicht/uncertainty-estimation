# https://www.tensorflow.org/tutorials/keras/save_and_load

import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from functions import CNN, get_data, CNN_transfer_learning

DATASET = "cifar100"
NUM_DATA = None
shape = (32, 32, 3)

train_images, train_labels, val_images, val_labels, test_images, test_labels, classes = get_data(DATASET, num_data=NUM_DATA)
filepath = "CNN_" + DATASET + ("" if NUM_DATA is None else "_" + str(NUM_DATA))

if DATASET == "cifar10":
    # transfer learning
    # first step: weights of main model frozen
    model = CNN_transfer_learning(path_pretrained_model="CNN_cifar100")
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3, restore_best_weights=True)
    model.fit(train_images, train_labels, batch_size=32, shuffle=True, epochs=1000,
              validation_data=(val_images, val_labels), callbacks=[early_stop])
    # if convergence: begin second step
    model.trainable = True
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

else:
    model = CNN(shape=shape, classes=classes)

# early stopping to monitor the validation loss and avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, restore_best_weights=True)
# reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)

model.fit(train_images, train_labels, batch_size=32, shuffle=True, epochs=1000,
          validation_data=(val_images, val_labels), callbacks=[early_stop, rlrop])

model.save(filepath)

# Create a basic model instance
model = CNN_transfer_learning(path_pretrained_model="CNN_cifar100") if DATASET == "cifar10" \
    else CNN(shape=shape, classes=classes)

# Evaluate the model
_, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

model = tf.keras.models.load_model(filepath)

# Re-evaluate the model
_, acc = model.evaluate(train_images, train_labels, verbose=2)
print("Restored model, Trainaccuracy: {:5.2f}%".format(100 * acc))

_, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
