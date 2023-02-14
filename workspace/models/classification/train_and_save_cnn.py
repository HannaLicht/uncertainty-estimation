# https://www.tensorflow.org/tutorials/keras/save_and_load

import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from functions import CNN, get_data

CONTINUE = False
DATASET = "cifar10"
NUM_DATA = 10000
shape = (32, 32, 3)

train_images, train_labels, val_images, val_labels, test_images, test_labels, classes, _, _ = \
    get_data(DATASET, num_data=NUM_DATA)
print(len(train_images))
print(len(val_images))

# Create a basic model instance
model = CNN(shape, classes)
model.summary()
if DATASET == "cifar10":
    model.load_weights("CNN_cifar100/cp.ckpt")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_path = "CNN_" + DATASET + ("" if NUM_DATA is None else "_" + str(NUM_DATA)) + "/cp.ckpt"

# early stopping to monitor the validation loss and avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, restore_best_weights=True)
# reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=1)
if CONTINUE:
    model.load_weights(checkpoint_path)

# Train the model with the new callback
model.fit(train_images,
          train_labels,
          batch_size=32,
          shuffle=True,
          epochs=1000,
          validation_data=(val_images, val_labels),
          callbacks=[cp_callback, early_stop, rlrop])  # Pass callback to training

# Create a basic model instance
model = CNN(shape, classes)

# Evaluate the model
_, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

model.load_weights(checkpoint_path)

# Re-evaluate the model
_, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
