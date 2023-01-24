import re
import keras.applications.efficientnet as efn
import sys
sys.path.append("/home/urz/hlichten")
from functions import CNN, get_train_and_test_data, build_effnet
from uncertainty.MC_Dropout import MCDropoutEstimator
import tensorflow as tf

T = 50
MODEL = "effnetb3"
CHECKPOINT_PATH = "../models/classification/" + MODEL + "/cp.ckpt"

if MODEL == "CNN_cifar100":
    _, _, x_val, y_val, x_test, y_test, num_classes = get_train_and_test_data("CNN_cifar100", validation_test_split=True)
    model = CNN(classes=100)
    model.load_weights(CHECKPOINT_PATH)
    estimator = MCDropoutEstimator(model, x_test, num_classes, T, xval=x_val, yval=y_val)
elif re.match("CNN_cifar10.*", MODEL):
    _, _, x_val, y_val, x_test, y_test, num_classes = get_train_and_test_data("CNN_cifar10", validation_test_split=True)
    model = CNN(classes=10)
    model.load_weights(CHECKPOINT_PATH)
    estimator = MCDropoutEstimator(model, x_test, num_classes, T, xval=x_val, yval=y_val)
elif MODEL == "effnetb3":
    _, _, x_val, y_val, x_test, y_test, num_classes = get_train_and_test_data("cars196", validation_test_split=True)
    eff = build_effnet(num_classes)
    eff.load_weights(CHECKPOINT_PATH)
    estimator = MCDropoutEstimator(eff, x_test, num_classes, T, xval=x_val, yval=y_val)
else:
    raise NotImplementedError

pred = estimator.get_ensemble_prediction()

print("Simple model accuracy:" + str(estimator.get_simple_model_accuracy(y_test)))
print("Ensemble accuracy: " + str(estimator.get_ensemble_accuracy(y_test)) + "\n")

y = tf.argmax(y_test, axis=-1).numpy()

print("UNCERTAINTY BY SHANNON ENTROPY")
uncertainties = estimator.uncertainties_shannon_entropy()
index = sorted(range(len(uncertainties)), key=uncertainties.__getitem__, reverse=False)
preds = [pred[i] for i in index]
lbls = [y[i] for i in index]
uncerts = [uncertainties[i] for i in index]
print("targets:     " + str(lbls[:10]) + " ... " + str(lbls[-10:]))
print("predictions: " + str(preds[:10]) + " ... " + str(preds[-10:]))
print("uncertainties: " + str(uncerts[:10]) + " ... " + str(uncerts[-10:]) + "\n")


print("UNCERTAINTY BY MUTUAL INFORMATION")
uncertainties = estimator.uncertainties_mutual_information()
index = sorted(range(len(uncertainties)), key=uncertainties.__getitem__, reverse=False)
uncerts = [uncertainties[i] for i in index]
preds = [pred[i] for i in index]
lbls = [y[i] for i in index]
print("targets:     " + str(lbls[:10]) + " ... " + str(lbls[-10:]))
print("predictions: " + str(preds[:10]) + " ... " + str(preds[-10:]))
print("uncertainties: " + str(uncerts[:10]) + " ... " + str(uncerts[-10:]) + "\n")

scores = estimator.certainty_scores(y)
print("score SE: " + str(scores[0]) + "     score MI: " + str(scores[1]))

estimator.plot_diagrams(y)
