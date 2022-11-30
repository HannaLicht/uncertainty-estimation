# https://www.tensorflow.org/tutorials/keras/save_and_load

import os
import tensorflow as tf
from functions import create_simple_model

(train_imgs, train_lbls), (test_imgs, test_lbls) = tf.keras.datasets.mnist.load_data()
train_imgs = train_imgs.reshape(-1, 28, 28, 1) / 255.0
test_imgs = test_imgs.reshape(-1, 28, 28, 1) / 255.0

# uncomment the following if you want to train only with one half of the data in order to retrain the model later
#train_imgs = train_imgs[:int(len(train_imgs)/2)]
#train_lbls = train_lbls[:int(len(train_lbls)/2)]

# Create a basic model instance
model = create_simple_model()

checkpoint_path = "simple_seq_model_mnist/cp.ckpt"

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_imgs,
          train_lbls,
          epochs=10,
          validation_data=(test_imgs, test_lbls),
          callbacks=[cp_callback])  # Pass callback to training


# Create a basic model instance
model = create_simple_model()

# Evaluate the model
loss, acc = model.evaluate(test_imgs, test_lbls, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = model.evaluate(test_imgs, test_lbls, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))