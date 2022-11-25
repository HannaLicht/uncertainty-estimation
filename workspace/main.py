import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


num_members = 5
(xtrain, _), _ = tf.keras.datasets.cifar10.load_data()
xtrain = xtrain.reshape(-1, 32, 32, 3) / 255.0
xtrain = xtrain[:100]

#data_augmentation = tf.keras.Sequential(
 #   [tf.keras.layers.RandomRotation(0.05),
  #  tf.keras.layers.RandomTranslation(0.1, 0.1),
   # tf.keras.layers.RandomZoom((0.0, 0.2), (0.0, 0.2))]
#)
data_augmentation = ImageDataGenerator(rotation_range=1, width_shift_range=0.05, height_shift_range=0.05,
                                       zoom_range=.1, horizontal_flip=True, fill_mode='reflect')

data_augmentation.fit(xtrain, augment=True)
aug_X_train = [data_augmentation.flow(xtrain, shuffle=False) for _ in range(num_members)]

plt.figure(figsize=(10, 10))
for i in range(9):
    augmented_image = data_augmentation.flow(xtrain, shuffle=False)
    plt.subplot(3, 3, i + 1)
    plt.imshow(next(augmented_image)[0])
    plt.axis("off")
plt.show()