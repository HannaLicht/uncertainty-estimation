import json
import re
import time
import sys
sys.path.append("/home/urz/hlichten")
from functions import CNN, get_data, build_effnet, CNN_transfer_learning
from uncertainty.MC_Dropout import MCDropoutEstimator
import tensorflow as tf

T = 50
MODEL = "CNN_cifar100"
MODEL_PATH = "../models/classification/" + MODEL
GET_TIMES = True


num_data = None
if re.match('CNN_cifar10_.*', MODEL):
    num_data = int(MODEL.replace('CNN_cifar10_', ""))

if re.match("CNN_cifar10.*", MODEL):
    data = "cifar100" if MODEL == "CNN_cifar100" else "cifar10"
    _, _, x_val, y_val, x_test, y_test, num_classes = get_data(data, num_data)
    model = CNN(classes=num_classes) if MODEL != "CNN_cifar10_100" else CNN_transfer_learning(num_classes)
elif MODEL == "effnetb3":
    _, _, x_val, y_val, x_test, y_test, num_classes = get_data("cars196")
    model = build_effnet(num_classes)
else:
    raise NotImplementedError

model = tf.keras.models.load_model(MODEL_PATH)

st = time.time()
estimator = MCDropoutEstimator(model, x_test, num_classes, T, xval=x_val, yval=y_val)
end = time.time()

if GET_TIMES:
    with open('../Results/times.json') as json_file:
        t = json.load(json_file)

    t[MODEL]["MC Dropout"]["with calibration"] = t[MODEL]["MC Dropout"]["with calibration"] + [round(end - st, 5)]

    model = tf.keras.models.load_model(MODEL_PATH)
    st = time.time()
    MCDropoutEstimator(model, x_test, num_classes, T)
    end = time.time()

    t[MODEL]["MC Dropout"]["uncertainty"] = t[MODEL]["MC Dropout"]["uncertainty"] + [round(end - st, 5)]

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
print("score SE: " + str(scores[0]) + "     score MI: " + str(scores[1]))

estimator.plot_diagrams(y)
