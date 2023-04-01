import json
import re
import time
from uncertainty.Ensemble import ENSEMBLE_LOCATION, BaggingEns, DataAugmentationEns, RandomInitShuffleEns
from functions import get_data, build_effnet, CNN, CNN_transfer_learning
import tensorflow as tf

model_name = "CNN_cifar10_100"
method = "bagging"
get_times = True

path_to_ensemble = ENSEMBLE_LOCATION + "/" + method + "/" + model_name

if method == "bagging":
    ens = BaggingEns
    key = "Bagging"
elif method == "data_augmentation":
    ens = DataAugmentationEns
    key = "Data Augmentation"
elif method == "rand_initialization_shuffle":
    ens = RandomInitShuffleEns
    key = "ZIS"
else:
    raise NotImplementedError

if model_name == "effnetb3":
    function = build_effnet
    data = "cars196"
elif model_name == "CNN_cifar100":
    function = CNN
    data = "cifar100"
else:
    function = CNN_transfer_learning
    data = "cifar10"

num_data = None
if re.match('CNN_cifar10_.*', model_name):
    num_data = int(model_name.replace('CNN_cifar10_', ""))

X_train, y_train, X_val, y_val, X_test, y_test, classes = get_data(data, num_data)

st = time.time()
estimator = ens(X_test, classes, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                build_model_function=function, path_to_ensemble=path_to_ensemble, val=True)
end = time.time()

if get_times:
    with open('../Results/times.json') as json_file:
        t = json.load(json_file)

    t[model_name][key]["with calibration"] = t[model_name][key]["with calibration"] + [round(end - st, 5)]

    st = time.time()
    ens(X_test, classes, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        build_model_function=function, path_to_ensemble=path_to_ensemble, val=False)
    end = time.time()

    t[model_name][key]["uncertainty"] = t[model_name][key]["uncertainty"] + [round(end - st, 5)]

    with open('../Results/times.json', 'w') as json_file:
        json.dump(t, json_file, indent=4)

pred = estimator.get_ensemble_prediction()

print("Simple model accuracy:" + str(estimator.get_simple_model_accuracy(y_test)))
print("Ensemble accuracy: " + str(estimator.get_ensemble_accuracy(y_test)) + "\n")
y_test = tf.argmax(y_test, axis=-1).numpy()

print("UNCERTAINTY BY SHANNON ENTROPY (predictive entropy)")
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
print("score PE: " + str(scores[0]) + "     score MI: " + str(scores[1]))

estimator.plot_diagrams(y_test)
