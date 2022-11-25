# https://towardsdatascience.com/cifar-100-transfer-learning-using-efficientnet-ed3ed7b89af2
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from functions import get_train_and_test_data, create_effnetb0_based_model, make_and_compile

height = 224
width = 224
channels = 3
input_shape = (height, width, channels)
checkpoint_path = "effnetb0/cp.ckpt"


def resize_img(img, shape):
    return tf.image.resize_with_pad(img, shape[1], shape[0])


xtrain, ytrain, xtest, ytest, _ = get_train_and_test_data("cifar100")
ytrain = tf.keras.utils.to_categorical(ytrain, 100)
ytest = tf.keras.utils.to_categorical(ytest, 100)

effnet, head = create_effnetb0_based_model(100)
model = make_and_compile(effnet, head)

#model.load_weights(checkpoint_path)
model.summary()

#early stopping to monitor the validation loss and avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
#reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr= 1e-6, verbose=1)


# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
model.fit(xtrain, ytrain, validation_data=(xtest, ytest),
          callbacks=[early_stop, cp_callback, rlrop], verbose=1, epochs=10)

_, acc = model.evaluate(xtest, ytest, verbose=2)
print("Validation accuracy: {:5.2f}%".format(100 * acc))