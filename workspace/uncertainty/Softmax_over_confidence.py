import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from metrics_classification import expected_calibration_error, static_calibration_error, reliability_diagram
from functions import create_simple_model

CHECKPOINT_PATH = "../models/classification/simple_seq_model_mnist/cp.ckpt"

"""Softmax Test"""

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(-1, 28, 28) / 255.0
test_images = test_images.reshape(-1, 28, 28) / 255.0

model = create_simple_model()
model.load_weights(CHECKPOINT_PATH)

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


outputs = model.predict(test_images)
print(expected_calibration_error(test_labels, outputs, 10))
print(expected_calibration_error(test_labels, outputs, 20))

print(static_calibration_error(test_labels, outputs, 10))
print(static_calibration_error(test_labels, outputs, 20))

reliability_diagram(test_labels, outputs, num_bins=10)
plt.show()

# trick model with OOD samples
rot_images = []
plt.figure(figsize=(12, 12))
for ind, img in enumerate(test_images[:16]):
    plt.subplot(4, 4, ind+1)
    img = img.reshape(28, 28, -1)
    rot_image = tf.image.rot90(img)
    plt.imshow(tf.reshape(rot_image, [28, 28]), vmin=0, vmax=1, cmap="Greys")
    plt.axis("off")
    rot_images.append(tf.reshape(rot_image, (28, 28)))
plt.show()

outputs = model.predict(tf.convert_to_tensor(rot_images))
softmax_out = tf.nn.softmax(outputs)
y_pred = np.argmax(softmax_out, axis=-1)
prob_y = tf.math.reduce_max(softmax_out, axis=-1)
for pred, prop, truth in zip(y_pred, prob_y, test_labels[:16]):
    print("prediction: ", pred, "   softmax: ", prop, "   before rotation: ", truth)

