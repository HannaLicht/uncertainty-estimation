import json
import sys
import re
import time

sys.path.append("/home/urz/hlichten")
from uncertainty.Ensemble import ENSEMBLE_LOCATION, BaggingEns, DataAugmentationEns, RandomInitShuffleEns
from functions import get_data, build_effnet, CNN, CNN_transfer_learning
import tensorflow as tf

NUM_MEMBERS = 5
MODEL = "effnetb3"
METHOD = "data_augmentation"
DATA = "cars196"
GET_TIMES = True
RUNS = 5

path_to_ensemble = ENSEMBLE_LOCATION + "/" + METHOD + "/" + MODEL

X_train, y_train, X_val, y_val, X_test, y_test, classes = get_data(DATA)

num_data = None
if re.match('CNN_cifar10_.*', MODEL):
    num_data = int(MODEL.replace('CNN_cifar10_', ""))

if METHOD == "bagging":
    ens = BaggingEns
    key = "Bagging"
elif METHOD == "data_augmentation":
    ens = DataAugmentationEns
    key = "Data Augmentation"
elif METHOD == "rand_initialization_shuffle":
    ens = RandomInitShuffleEns
    key = "ZIS"
else:
    raise NotImplementedError

if MODEL == "effnetb3":
    function = build_effnet
elif MODEL == "CNN_cifar10_100":
    function = CNN_transfer_learning
else:
    function = CNN

for _ in range(RUNS):
    st = time.time()
    estimator = ens(X_test, classes, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                    build_model_function=function, path_to_ensemble=path_to_ensemble, num_members=NUM_MEMBERS, val=True)
    end = time.time()

    if GET_TIMES:
        with open('../Results/times.json') as json_file:
            t = json.load(json_file)

        t[MODEL][key]["with calibration"] = t[MODEL][key]["with calibration"] + [round(end - st, 5)]

        st = time.time()
        ens(X_test, classes, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            build_model_function=function, path_to_ensemble=path_to_ensemble, num_members=NUM_MEMBERS, val=False)
        end = time.time()

        t[MODEL][key]["uncertainty"] = t[MODEL][key]["uncertainty"] + [round(end - st, 5)]

        if MODEL != "effnetb3":
            st = time.time()
            ens(X_test, classes, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                build_model_function=function, num_members=NUM_MEMBERS, val=False)
            end = time.time()
            t[MODEL][key]["preparation & uncertainty"] = t[MODEL][key]["preparation & uncertainty"] + [round(end - st, 5)]

        with open('../Results/times.json', 'w') as json_file:
            json.dump(t, json_file, indent=4)

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
