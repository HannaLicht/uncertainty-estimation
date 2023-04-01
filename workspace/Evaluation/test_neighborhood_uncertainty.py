import json
import re
import time
from matplotlib import pyplot as plt
from uncertainty.calibration_classification import expected_calibration_error
from functions import CNN, get_data, split_validation_from_train, build_effnet, CNN_transfer_learning
import tensorflow as tf
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier

runs = 4
use_validation_data = False
use_stored = False       # if you want to use a stored certainty network

model_name = "CNN_cifar10_100"
data = "cifar10"

assert (use_validation_data and model_name != "CNN_cifar10_100") or not use_validation_data

num_data = None
if re.match('CNN_cifar10_.*', model_name):
    num_data = int(model_name.replace('CNN_cifar10_', ""))

model_path = "../models/classification/" + model_name
xtrain, ytrain, xval, yval, xtest, ytest, cl = get_data(data, num_data)

model = tf.keras.models.load_model(model_path)
model.evaluate(xtrain, ytrain)
model.evaluate(xtest, ytest)

ypred = tf.argmax(model.predict(xtest), axis=-1).numpy()

if use_validation_data:
    method = "NUC Va"
    pre_path_certainty_model = "../models/classification/certainty_model/val/3/"
    if model_name != "effnetb3":
        x_train, y_train = xval[:int(4*len(xval) / 5)], yval[:int(4*len(yval) / 5)]
        x_val, y_val = xval[int(4*len(xval) / 5):], yval[int(4*len(yval) / 5):]
    else:
        x_train, y_train, x_val, y_val = split_validation_from_train(xval, yval, cl, num_imgs_per_class=2)
else:
    x_train, y_train, x_val, y_val = xtrain, ytrain, xval, yval
    method = "NUC Tr"
    pre_path_certainty_model = "../models/classification/certainty_model/train/3/"

path_certainty_model = pre_path_certainty_model + model_name + "/cp.ckpt"
model.evaluate(xtest, ytest)

estimator = NeighborhoodUncertaintyClassifier(model, x_train, y_train, x_val, y_val, xtest,
                                              path_certainty_model=path_certainty_model)

certainties = estimator.certainties
y_lbls = tf.argmax(ytest, axis=-1)
index = sorted(range(len(certainties)), key=certainties.__getitem__, reverse=False)
print("most and least uncertain samples: " + str(index[0]) + ": " + str(certainties[index[0]]) + ",   " +
      str(index[-1]) + ": " + str(certainties[index[-1]]))
ypred = tf.argmax(model.predict(xtest), axis=-1).numpy()
preds = [ypred[i] for i in index]
lbls = [y_lbls.numpy()[i] for i in index]
certs = [certainties[i] for i in index]
print("targets:     " + str(lbls[:10]) + " ... " + str(lbls[-10:]))
print("predictions: " + str(preds[:10]) + " ... " + str(preds[-10:]))
print("certainties: " + str(certs[:10]) + " ... " + str(certs[-10:]))

estimator.plot_diagrams(y_lbls)
print("score = ", estimator.certainty_score(y_lbls))


# Test: which k is best -> auroc, aupr
incorrect = (tf.argmax(ytest, axis=-1) != ypred)
last_k = 99 if model_name == "CNN_cifar10_100" else 100

for _ in range(runs):
    method = "NUC Va"
    pre_path_certainty_model = "../models/classification/certainty_model/val/"
    if model_name != "effnetb3":
        x_train, y_train = xval[:int(4 * len(xval) / 5)], yval[:int(4 * len(yval) / 5)]
        x_val, y_val = xval[int(4 * len(xval) / 5):], yval[int(4 * len(yval) / 5):]
    else:
        x_train, y_train, x_val, y_val = split_validation_from_train(xval, yval, cl, num_imgs_per_class=2)

    for valid in [True, False]:
        if model_name == "CNN_cifar10_100" and valid:
            method = "NUC Tr"
            pre_path_certainty_model = "../models/classification/certainty_model/train/"
            x_train, y_train, x_val, y_val = xtrain, ytrain, xval, yval
            continue

        for count, k in enumerate([3, 5, 10, 25, 50, last_k]):
            path_certainty_model = None
            if use_stored:
                path_certainty_model = pre_path_certainty_model + str(k) + "/" + model_name +"/cp.ckpt"

            st = time.time()
            estimator = NeighborhoodUncertaintyClassifier(model, x_train, y_train, x_val, y_val, xtest,
                                                          path_certainty_model=path_certainty_model, k=k)
            end = time.time()

            with open('../Results/times.json') as json_file:
                t = json.load(json_file)

            if use_stored:
                t[model_name][method]["uncertainty"][count] = t[model_name][method]["uncertainty"][count] + \
                                                              [round(end - st, 5)]
                with open('../Results/times.json', 'w') as json_file:
                    json.dump(t, json_file, indent=4)

            else:
                t[model_name][method]["preparation & uncertainty"][count] = t[model_name][method][
                                                                         "preparation & uncertainty"][count] + \
                                                                            [round(end - st, 5)]
                with open('../Results/times.json', 'w') as json_file:
                    json.dump(t, json_file, indent=4)

                m = tf.keras.metrics.AUC(curve='ROC')
                m.update_state(incorrect, 1 - estimator.certainties)
                auroc = m.result().numpy()

                m = tf.keras.metrics.AUC(curve="PR")
                m.update_state(incorrect, 1 - estimator.certainties)
                aupr = m.result().numpy()

                with open('../Results/auroc_aupr.json') as json_file:
                    data = json.load(json_file)
                data[method][model_name]["auroc"][count] = data[method][model_name]["auroc"][count] + [auroc.item()]
                data[method][model_name]["aupr"][count] = data[method][model_name]["aupr"][count] + [aupr.item()]
                with open('../Results/auroc_aupr.json', 'w') as json_file:
                    json.dump(data, json_file, indent=4)

                with open('../Results/eces.json') as json_file:
                    data = json.load(json_file)
                ece = expected_calibration_error(tf.argmax(ytest, axis=-1), ypred, estimator.certainties).numpy()
                data[model_name][method][count] = data[model_name][method][count] + [ece.item()]
                with open('../Results/eces.json', 'w') as json_file:
                    json.dump(data, json_file, indent=4)

        method = "NUC Tr"
        pre_path_certainty_model = "../models/classification/certainty_model/train/"
        x_train, y_train, x_val, y_val = xtrain, ytrain, xval, yval

