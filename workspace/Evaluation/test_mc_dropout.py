import json
import re
import time
from functions import CNN, get_data, build_effnet, CNN_transfer_learning
from uncertainty.MC_Dropout import MCDropoutEstimator
import tensorflow as tf

T = 50
model_name = "CNN_cifar10_100"
model_path = "../models/classification/" + model_name
get_times = False


num_data = None
if re.match('CNN_cifar10_.*', model_name):
    num_data = int(model_name.replace('CNN_cifar10_', ""))

if re.match("CNN_cifar10.*", model_name):
    data = "cifar100" if model_name == "CNN_cifar100" else "cifar10"
    _, _, x_val, y_val, x_test, y_test, num_classes = get_data(data, num_data)
    model = CNN(classes=num_classes) if model_name == "CNN_cifar100" else CNN_transfer_learning(num_classes)
elif model_name == "effnetb3":
    _, _, x_val, y_val, x_test, y_test, num_classes = get_data("cars196")
    model = build_effnet(num_classes)
else:
    raise NotImplementedError

model = tf.keras.models.load_model(model_path)

st = time.time()
estimator = MCDropoutEstimator(model, x_test, num_classes, T, xval=x_val, yval=y_val)
end = time.time()

if get_times:
    with open('../Results/times.json') as json_file:
        t = json.load(json_file)

    t[model_name]["MC Dropout"]["with calibration"] = t[model_name]["MC Dropout"]["with calibration"] + [round(end - st, 5)]

    model = tf.keras.models.load_model(model_path)
    st = time.time()
    MCDropoutEstimator(model, x_test, num_classes, T)
    end = time.time()

    t[model_name]["MC Dropout"]["uncertainty"] = t[model_name]["MC Dropout"]["uncertainty"] + [round(end - st, 5)]

    with open('../Results/times.json', 'w') as json_file:
        json.dump(t, json_file, indent=4)

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
print("score PE: " + str(scores[0]) + "     score MI: " + str(scores[1]))

estimator.plot_diagrams(y)
