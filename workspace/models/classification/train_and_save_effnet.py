# https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
import sys
sys.path.append("/home/urz/hlichten")
from functions import get_train_and_test_data, build_effnet
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

batch_size = 32

xtrain, ytrain, xval, yval, xtest, ytest, num_classes = get_train_and_test_data("cars196", True)

model = build_effnet(num_classes=num_classes)

# early stopping to monitor the validation loss and avoid overfitting
early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=15, restore_best_weights=True)
# reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="effnetb3/cp.ckpt",
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=1)

model.fit(xtrain, ytrain, validation_data=(xval, yval), epochs=1000,
          callbacks=[early_stop, cp_callback, rlrop], shuffle=True, batch_size=batch_size)

_, acc = model.evaluate(xtest, ytest, verbose=2)
print("Test accuracy: {:5.2f}%".format(100 * acc))
