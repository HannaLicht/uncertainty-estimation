import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from calibration_classification import expected_calibration_error, static_calibration_error, reliability_diagram
from functions import CNN, get_train_and_test_data

"""Softmax Test"""

train_images, train_labels, val_images, val_labels, test_images, test_labels, classes = \
    get_train_and_test_data("mnist", validation_test_split=True)

model = tf.keras.models.Sequential([tf.keras.layers.Input((28, 28, 1)),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation="relu"),
                                    tf.keras.layers.Dense(128, activation="relu"),
                                    tf.keras.layers.Dense(10, activation="softmax")])
model.summary()
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, tf.one_hot(train_labels, 10), validation_data=(val_images, tf.one_hot(val_labels, 10)),
          epochs=5)

loss, acc = model.evaluate(test_images, tf.one_hot(test_labels, 10), verbose=2)
print("Test accuracy: {:5.2f}%".format(100 * acc))

#test_labels = np.argmax(test_labels, axis=-1)
outputs = model.predict(test_images)

print(expected_calibration_error(test_labels, outputs, certainties=np.max(outputs, axis=-1), num_bins=10))
print(expected_calibration_error(test_labels, outputs, certainties=np.max(outputs, axis=-1), num_bins=20))

print(static_calibration_error(test_labels, outputs, num_bins=10))
print(static_calibration_error(test_labels, outputs, num_bins=20))

reliability_diagram(test_labels, outputs, num_bins=10)
plt.show()


# trick with OOD examples:

data_augmentation = ImageDataGenerator(rotation_range=100, horizontal_flip=True, vertical_flip=True, fill_mode='reflect')
data_augmentation.fit(test_images, augment=True)
iterator = data_augmentation.flow(test_images, shuffle=True, batch_size=128)

plt.figure(figsize=(9, 4.5))
for i in range(8):
    augmented_batch = next(iterator)
    plt.subplot(2, 4, i + 1)
    plt.imshow(augmented_batch[0])
    out = model.predict(augmented_batch)[0]
    pred = tf.argmax(out).numpy()
    certainty = tf.reduce_max(out).numpy()
    plt.title("Präd: " + str(pred) + "  Cert: " + str(round(certainty*100.)/100.))
    plt.axis("off")
plt.show()
