import sys
import re
sys.path.append("/home/urz/hlichten")
from uncertainty.Ensemble import ENSEMBLE_LOCATION, BaggingEns, DataAugmentationEns, RandomInitShuffleEns
from functions import get_train_and_test_data
import tensorflow as tf

NUM_MEMBERS = 5
MODEL = "effnetb3"
METHOD = "rand_initialization_shuffle"
DATA = "cars196"

path_to_ensemble = ENSEMBLE_LOCATION + "/" + METHOD + "/" + MODEL

X_train, y_train, X_val, y_val, X_test, y_test, classes = get_train_and_test_data(DATA, validation_test_split=True)

num_data = None
if re.match('CNN_cifar10_.*', MODEL):
    num_data = int(MODEL.replace('CNN_cifar10_', ""))
    MODEL = "CNN_cifar10"
    X_train = X_train[:num_data]
    y_train = y_train[:num_data]

if METHOD == "bagging":
    estimator = BaggingEns(X_test, classes, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                           model_name=MODEL, path_to_ensemble=path_to_ensemble, num_members=NUM_MEMBERS, val=True)
elif METHOD == "data_augmentation":
    estimator = DataAugmentationEns(X_test, classes, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                           model_name=MODEL, path_to_ensemble=path_to_ensemble, num_members=NUM_MEMBERS, val=True)
elif METHOD == "rand_initialization_shuffle":
    estimator = RandomInitShuffleEns(X_test, classes, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                           model_name=MODEL, path_to_ensemble=path_to_ensemble, num_members=NUM_MEMBERS, val=True)
else:
    raise NotImplementedError

pred = estimator.get_ensemble_prediction()

print("Simple model accuracy:" + str(estimator.get_simple_model_accuracy(y_test)))
print("Ensemble accuracy: " + str(estimator.get_ensemble_accuracy(y_test)) + "\n")
y_test = tf.argmax(y_test, axis=-1).numpy()

print("UNCERTAINTY BY SHANNON ENTROPY")
uncertainties = estimator.uncertainties_shannon_entropy()
index = sorted(range(uncertainties.shape[0]), key=uncertainties.__getitem__, reverse=False)
preds = [pred[i] for i in index]
lbls = [y_test[i] for i in index]
uncerts = [uncertainties[i] for i in index]
print("targets:     " + str(lbls[:10]) + " ... " + str(lbls[-10:]))
print("predictions: " + str(preds[:10]) + " ... " + str(preds[-10:]))
print("uncertainties: " + str(uncerts[:10]) + " ... " + str(uncerts[-10:]) + "\n")


print("UNCERTAINTY BY MUTUAL INFORMATION")
uncertainties = estimator.uncertainties_mutual_information()
index = sorted(range(uncertainties.shape[0]), key=uncertainties.__getitem__, reverse=False)
uncerts = [uncertainties[i] for i in index]
preds = [pred[i] for i in index]
lbls = [y_test[i] for i in index]
print("targets:     " + str(lbls[:10]) + " ... " + str(lbls[-10:]))
print("predictions: " + str(preds[:10]) + " ... " + str(preds[-10:]))
print("uncertainties: " + str(uncerts[:10]) + " ... " + str(uncerts[-10:]) + "\n")

scores = estimator.certainty_scores(y_test)
print("score SE: " + str(scores[0]) + "     score MI: " + str(scores[1]))

estimator.plot_diagrams(y_test)
