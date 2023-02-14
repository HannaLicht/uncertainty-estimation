import json
import re
import sys
import time

from matplotlib import pyplot as plt

sys.path.append("/home/urz/hlichten")
from functions import CNN, get_data, split_validation_from_train, build_effnet
import tensorflow as tf
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier

RUNS = 9
SAVE_OR_USE_SAVED = False
GET_TIMES = False

model_name = "CNN_cifar10_10000"
data = "cifar10"
use_validation_data = False

assert (use_validation_data and model_name != "CNN_cifar10_100")\
       or not use_validation_data

num_data = None
if re.match('CNN_cifar10_.*', model_name):
    num_data = int(model_name.replace('CNN_cifar10_', ""))

checkpoint_path = "../models/classification/" + model_name + "/cp.ckpt"
xtrain, ytrain, xval, yval, xtest, ytest, cl, _, _ = get_data(data, num_data)


'''def augment_images(x, y):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomTranslation([-0.1, 0.1], [-0.1, 0.1])
    ])
    x_new, y_new = list(x), list(y)
    for img, lbl in zip(x, y):
        x_new = x_new + [data_augmentation(img) for _ in range(4)]
        y_new = y_new + [lbl for _ in range(4)]

    plt.figure(figsize=(10, 10))
    plt.subplot(3, 3, 1)
    plt.imshow(x[0]/255.)
    plt.axis("off")
    for i in range(8):
        img = data_augmentation(x[0])
        plt.subplot(3, 3, i + 2)
        plt.imshow(img/225.)
        plt.axis("off")
    plt.savefig("../plots/test_augmentation.png")
    plt.show()

    return tf.reshape(x_new, (-1, 300, 300, 3)), tf.reshape(y_new, (-1, 196))'''


if use_validation_data:
    method = "NUC Validation"
    pre_path_uncertainty_model = "../models/classification/uncertainty_model/val/3/"
    if model_name != "effnetb3":
        xtrain, ytrain = xval[:int(4*len(xval) / 5)], yval[:int(4*len(yval) / 5)]
        xval, yval = xval[int(4*len(xval) / 5):], yval[int(4*len(yval) / 5):]
    else:
        xtrain, ytrain, xval, yval = split_validation_from_train(xval, yval, cl, num_imgs_per_class=2)
        '''if augment_data:
            xtrain, ytrain = augment_images(xtrain, ytrain)
            xval, yval = augment_images(xval, yval)'''
else:
    method = "NUC Training"
    pre_path_uncertainty_model = "../models/classification/uncertainty_model/train/3/"

path_uncertainty_model = pre_path_uncertainty_model + model_name + "/cp.ckpt"
model = CNN(classes=100 if model_name == "CNN_cifar100" else 10) if model_name != "effnetb3" else build_effnet(cl)
model.load_weights(checkpoint_path)


model.evaluate(xtest, ytest)

st = time.time()
estimator = NeighborhoodUncertaintyClassifier(model, xtrain, ytrain, xval, yval, xtest,
                                              path_uncertainty_model=path_uncertainty_model)
end = time.time()

if GET_TIMES:
    with open('../Results/times.json') as json_file:
        t = json.load(json_file)

    t[model_name][method]["uncertainty"] = t[model_name][method]["uncertainty"] + [round(end - st, 5)]

    st = time.time()
    NeighborhoodUncertaintyClassifier(model, xtrain, ytrain, xval, yval, xtest)
    end = time.time()

    t[model_name][method]["preparation & uncertainty"] = t[model_name][method]["preparation & uncertainty"] + [round(end - st, 5)]

    with open('../Results/times.json', 'w') as json_file:
        json.dump(t, json_file, indent=4)

    quit()

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
last_k = 99 if model_name == "CNN_cifar10_100" else 100

for _ in range(RUNS):
    auroc, aupr = [], []

    for k in [3, 5, 10, 25, 50, last_k]:
        path_uncertainty_model = None
        if SAVE_OR_USE_SAVED:
            path_uncertainty_model = pre_path_uncertainty_model + str(k) + "/" + model_name +"/cp.ckpt"

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
        with open('../Results/auroc_aupr.json') as json_file:
            data = json.load(json_file)
            data[name_method][model_name]["auroc"][i] = data[name_method][model_name]["auroc"][i] + [roc.item()]
            data[name_method][model_name]["aupr"][i] = data[name_method][model_name]["aupr"][i] + [pr.item()]
        with open('../Results/auroc_aupr.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)



