import json
import re
import sys
sys.path.append("/home/urz/hlichten")
from functions import CNN, get_train_and_test_data, split_validation_from_train, build_effnet
import tensorflow as tf
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier

RUNS = 4
SAVE_OR_USE_SAVED = False

model_name = "effnetb3"
data = "cars196"
use_validation_data = True

checkpoint_path = "../models/classification/" + model_name + "/cp.ckpt"
xtrain, ytrain, xval, yval, xtest, ytest, cl = get_train_and_test_data(data, validation_test_split=True)

if use_validation_data:
    pre_path_uncertainty_model = "../models/classification/uncertainty_model/"
    if model_name != "effnetb3":
        xtrain, ytrain = xval[:int(4*len(xval) / 5)], yval[:int(4*len(yval) / 5)]
        xval, yval = xval[int(4*len(xval) / 5):], yval[int(4*len(yval) / 5):]
    else:
        xtrain, ytrain, xval, yval = split_validation_from_train(xval, yval, cl, num_imgs_per_class=2)
else:
    pre_path_uncertainty_model = "../models/classification/uncertainty_model/trained_on_traindata/"

path_uncertainty_model = pre_path_uncertainty_model + model_name + "/cp.ckpt"
model = CNN(classes=100 if model_name == "CNN_cifar100" else 10) if model_name != "effnetb3" else build_effnet(cl)
model.load_weights(checkpoint_path)

num_data = None
if re.match('CNN_cifar10_.*', model_name) and not use_validation_data:
    num_data = int(model_name.replace('CNN_cifar10_', ""))
    xtrain, ytrain = xtrain[:num_data], ytrain[:num_data]


model.evaluate(xtest, ytest)

estimator = NeighborhoodUncertaintyClassifier(model, xtrain, ytrain, xval, yval, xtest,
                                              path_uncertainty_model=path_uncertainty_model)
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
incorrect = (tf.argmax(ytest, axis=-1) != tf.argmax(model.predict(xtest), axis=-1).numpy())
name_method = "nuc_val" if use_validation_data else "nuc_train"

for _ in range(RUNS):
    auroc, aupr = [], []

    for k in [5, 10, 25, 50, 100]:
        path_uncertainty_model = None
        if SAVE_OR_USE_SAVED:
            path_uncertainty_model = pre_path_uncertainty_model + "different_k/" + str(k) + "/" + model_name +"/cp.ckpt"
            if k == 10:
                path_uncertainty_model = pre_path_uncertainty_model + model_name + "/cp.ckpt"

        estimator = NeighborhoodUncertaintyClassifier(model, xtrain, ytrain, xval, yval, xtest,
                                                      path_uncertainty_model=path_uncertainty_model, k=k)
        m = tf.keras.metrics.AUC(curve='ROC')
        m.update_state(incorrect, 1-estimator.certainties)
        auroc.append(m.result().numpy())

        m = tf.keras.metrics.AUC(curve="PR")
        m.update_state(incorrect, 1-estimator.certainties)
        aupr.append(m.result().numpy())

    print("AUROCs: ", auroc)
    print("AUPRs: ", aupr)

    for i, (roc, pr) in enumerate(zip(auroc, aupr)):
        with open('results_auroc_aupr.json') as json_file:
            data = json.load(json_file)
            data[name_method][model_name]["auroc"][i] = data[name_method][model_name]["auroc"][i] + [roc.item()]
            data[name_method][model_name]["aupr"][i] = data[name_method][model_name]["aupr"][i] + [pr.item()]
        with open('results_auroc_aupr.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)



