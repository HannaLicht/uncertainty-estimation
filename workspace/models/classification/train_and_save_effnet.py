# https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
from functions import get_data, build_effnet
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

batch_size = 32

xtrain, ytrain, xval, yval, xtest, ytest, num_classes = get_data("cars196")

# transfer learning
# first step: weights of main model frozen
model = build_effnet(num_classes=num_classes)
early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=3, restore_best_weights=True)
model.fit(xtrain, ytrain, batch_size=32, shuffle=True, epochs=1000, validation_data=(xval, yval),
          callbacks=[early_stop])

# if convergence: begin second step
model.trainable = True
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# early stopping to monitor the validation loss and avoid overfitting
early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=15, restore_best_weights=True)
# reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)

# Create a callback that saves the model's weights

model.fit(xtrain, ytrain, validation_data=(xval, yval), epochs=1000,
          callbacks=[early_stop, rlrop], shuffle=True, batch_size=batch_size)
model.save("effnetb3")

_, acc = model.evaluate(xtrain, ytrain, verbose=2)
print("Test accuracy: {:5.2f}%".format(100 * acc))

_, acc = model.evaluate(xtest, ytest, verbose=2)
print("Test accuracy: {:5.2f}%".format(100 * acc))
