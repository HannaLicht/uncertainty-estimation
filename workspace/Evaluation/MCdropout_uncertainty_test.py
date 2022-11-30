import re

import keras.applications.efficientnet as efn
from matplotlib import pyplot as plt

from functions import create_simple_model, get_test_data, ResNet
from uncertainty.MC_Dropout import MCDropoutEstimator
import tensorflow as tf

T = 50
MODEL = "ResNet_cifar10"
CHECKPOINT_PATH = "../models/classification/" + MODEL + "/cp.ckpt"

if MODEL == "simple_seq_model_mnist":
    x, y, num_classes = get_test_data("mnist")
    model = create_simple_model()
    model.load_weights(CHECKPOINT_PATH)
    estimator = MCDropoutEstimator(model, x, num_classes, T)
elif MODEL == "simple_seq_model_fashion_mnist":
    x, y, num_classes = get_test_data("fashion_mnist")
    model = create_simple_model()
    model.load_weights(CHECKPOINT_PATH)
    estimator = MCDropoutEstimator(model, x, num_classes, T)
elif MODEL == "ResNet_cifar100":
    x, y, num_classes = get_test_data("cifar100")
    model = ResNet(classes=100)
    model.load_weights(CHECKPOINT_PATH)
    estimator = MCDropoutEstimator(model, x, num_classes, T)
elif re.match("ResNet_cifar10.*", MODEL):
    x, y, num_classes = get_test_data("cifar10")
    model = ResNet(classes=10)
    model.load_weights(CHECKPOINT_PATH)
    estimator = MCDropoutEstimator(model, x, num_classes, T)
elif MODEL == "effnetb0":
    x, y, num_classes = get_test_data("imagenette")
    eff = efn.EfficientNetB0(weights='imagenet')
    estimator = MCDropoutEstimator(eff, x, num_classes, T)
else:
    raise NotImplementedError

pred = estimator.get_ensemble_prediction()

print("Simple model accuracy:" + str(estimator.get_simple_model_accuracy(y)))
print("Ensemble accuracy: " + str(estimator.get_ensemble_accuracy(y)) + "\n")

y = tf.argmax(y, axis=-1).numpy()

print("UNCERTAINTY BY SHANNON ENTROPY")
certainties = estimator.bounded_certainties_shannon_entropy()
index = sorted(range(len(certainties)), key=certainties.__getitem__, reverse=False)
preds = [pred[i] for i in index]
lbls = [y[i] for i in index]
certs = [certainties[i] for i in index]
print("targets:     " + str(lbls[:10]) + " ... " + str(lbls[-10:]))
print("predictions: " + str(preds[:10]) + " ... " + str(preds[-10:]))
print("certainties: " + str(certs[:10]) + " ... " + str(certs[-10:]) + "\n")


print("UNCERTAINTY BY MUTUAL INFORMATION")
certainties = estimator.bounded_certainties_mutual_information()
index = sorted(range(len(certainties)), key=certainties.__getitem__, reverse=False)
certs = [certainties[i] for i in index]
preds = [pred[i] for i in index]
lbls = [y[i] for i in index]
print("targets:     " + str(lbls[:10]) + " ... " + str(lbls[-10:]))
print("predictions: " + str(preds[:10]) + " ... " + str(preds[-10:]))
print("certainties: " + str(certs[:10]) + " ... " + str(certs[-10:]) + "\n")

scores = estimator.certainty_scores(y)
print("score SE: " + str(scores[0]) + "     score MI: " + str(scores[1]))

estimator.plot_diagrams(y)
plt.show()
