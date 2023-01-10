import tensorflow as tf
import sys
sys.path.append("/home/urz/hlichten")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

from functions import get_train_and_test_data, CNN
from uncertainty.Ensemble import BaggingEns, DataAugmentationEns, RandomInitShuffleEns
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
from uncertainty.calibration_classification import expected_calibration_error, get_normalized_certainties

tfd = tfp.distributions
'''

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


x = [5, 10, 25, 50, 100]

c10_100 = [0.73482597, 0.7465769, 0.75602734, 0.762831, 0.76554745]
c10_1000 = [0.7392374, 0.7450949, 0.7495748, 0.75054413, 0.7502146]
c10_10000 = [0.780, 0.778, 0.773, 0.772, 0.769]
c10 = [0.825, 0.823, 0.808, 0.804, 0.805]
c100 = [0.80399644, 0.8025899, 0.7999729, 0.7871314, 0.7688228]
imgnet = []

plt.figure(figsize=(11, 4.5))
plt.subplot(1, 2, 1)
plt.ylabel("AUROC")
plt.xlabel("Anzahl Nachbarn (k)")
plt.plot(x, c10_100, label="cifar10 (100 data)", marker='.', color=adjust_lightness('b', 1.6))
plt.plot(x, c10_1000, label="cifar10 (1000 data)", marker='.', color=adjust_lightness('b', 1.3))
plt.plot(x, c10_10000, label="cifar10 (10000 data)", marker='.', color=adjust_lightness('b', 0.8))
plt.plot(x, c10, label="cifar10", marker='.', color=adjust_lightness('b', 0.4))
plt.plot(x, c100, label="cifar100", marker='.', color='tomato')
#plt.plot(x, imgnet, label="imagenet", marker='.', color='yellowgreen')

box = plt.subplot(1, 2, 1).get_position()
plt.subplot(1, 2, 1).set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

c10_100 = [0.58416265, 0.602342, 0.60948586, 0.6230147, 0.62302554]
c10_1000 = [0.7557357, 0.76433206, 0.7682079, 0.77033293, 0.7690269]
c10_10000 = [0.867, 0.863, 0.861, 0.863, 0.862]
c10 = [0.926, 0.926, 0.918, 0.917, 0.918]
c100 = [0.7694196, 0.76708686, 0.7638339, 0.7492295, 0.7310469]
imgnet = []

plt.subplot(1, 2, 2)
plt.ylabel("AUPR")
plt.xlabel("Anzahl Nachbarn (k)")
plt.plot(x, c10_100, label="cifar10 (100 data)", marker='.', color=adjust_lightness('b', 1.6))
plt.plot(x, c10_1000, label="cifar10 (1000 data)", marker='.', color=adjust_lightness('b', 1.3))
plt.plot(x, c10_10000, label="cifar10 (10000 data)", marker='.', color=adjust_lightness('b', 0.8))
plt.plot(x, c10, label="cifar10", marker='.', color=adjust_lightness('b', 0.4))
plt.plot(x, c100, label="cifar100", marker='.', color='tomato')
#plt.plot(x, imgnet, label="imagenet", marker='.', color='yellowgreen')

box = plt.subplot(1, 2, 2).get_position()
plt.subplot(1, 2, 2).set_position([box.x0*0.9, box.y0, box.width * 0.8, box.height*0.8])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
'''

'''
# calibration plot
plt.figure(figsize=(4.5, 3.7))
plt.plot([0, 1], [0, 1], 'k--', label="Perfekt kalibriert")
plt.plot(np.linspace(start=0, stop=0.5, num=50), tf.sigmoid(np.linspace(start=-6, stop=0, num=50)),
         label="Zu selbstsicher", color="red")
plt.plot(np.linspace(start=0.5, stop=1, num=50), tf.sigmoid(np.linspace(start=0, stop=6, num=50)),
         label="Zu unsicher", color="brown")
plt.fill_between(np.linspace(start=0, stop=0.5, num=50), np.linspace(start=0, stop=0.5, num=50),
                 tf.sigmoid(np.linspace(start=-6, stop=0, num=50)), color='red', alpha=0.15)
plt.fill_between(np.linspace(start=0.5, stop=1, num=50), np.linspace(start=0.5, stop=1, num=50),
                 tf.sigmoid(np.linspace(start=0, stop=6, num=50)), color='brown', alpha=0.15)
plt.xlabel('Konfidenz')
plt.ylabel('tatsächliche Accuracy')
plt.legend(loc="upper left")
plt.savefig("Evaluation/plots/calibration_plot.png", dpi=300)
plt.show()
'''

method = "mc_drop"

for model_name in ["CNN_cifar10_100", "CNN_cifar10_1000", "CNN_cifar10_10000", "CNN_cifar10", "CNN_cifar100"]:
    xtrain, ytrain, xval, yval, xtest, ytest, cl = get_train_and_test_data("cifar10" if model_name != "CNN_cifar100"
                                                                           else "cifar100",
                                                                           validation_test_split=True)

    model = CNN(classes=cl)
    model.load_weights("models/classification/" + model_name + "/cp.ckpt")
    ypred = model.predict(xtest)

    path = "models/classification/ensembles/" + method + "/" + model_name
    if method == "mc_drop":
        estimator = MCDropoutEstimator(model, xtest, cl, xval=xval, yval=yval)
        ypred = estimator.p_ens
    elif method == "softmax":
        soft_ent_uncert_val = tfd.Categorical(probs=model.predict(xval, verbose=0)).entropy().numpy()
        soft_ent_uncert_test = tfd.Categorical(probs=model.predict(xtest, verbose=0)).entropy().numpy()
        softmax_entropy = get_normalized_certainties(model.predict(xval, verbose=0), yval,
                                                     soft_ent_uncert_val, soft_ent_uncert_test)
        ece = expected_calibration_error(tf.argmax(ytest, axis=-1), ypred, softmax_entropy).numpy()
        print(model_name + " ", ece)
        continue
    elif method == "bagging":
        estimator = BaggingEns(xtrain, ytrain, xtest, cl, model_name, path, xval, yval, val=True)
        ypred = estimator.p_ens
    elif method == "data_augmentation":
        estimator = DataAugmentationEns(xtrain, ytrain, xtest, cl, model_name, path, xval, yval, val=True)
        ypred = estimator.p_ens
    elif method == "rand_initialization_shuffle":
        estimator = RandomInitShuffleEns(xtrain, ytrain, xtest, cl, model_name, path, xval, yval, val=True)
        ypred = estimator.p_ens
    elif method == "nuc":
        path = "models/classification/uncertainty_model/"
        xtrain, ytrain = xval[:int(len(xval) / 2)], yval[:int(len(yval) / 2)]
        xval, yval = xval[int(len(xval) / 2):], yval[int(len(yval) / 2):]
        estimator = NeighborhoodUncertaintyClassifier(model, xtrain, ytrain, xval, yval, xtest,
                                                  path + model_name.replace('CNN_', "") + "/cp.ckpt")
        ece = expected_calibration_error(tf.argmax(ytest, axis=-1), ypred, estimator.certainties).numpy()
        print(model_name + " ", ece)
        continue
    else:
        raise NotImplementedError

    ece_se = expected_calibration_error(tf.argmax(ytest, axis=-1), ypred,
                                        estimator.normalized_certainties_shannon_entropy()).numpy()
    ece_mi = expected_calibration_error(tf.argmax(ytest, axis=-1), ypred,
                                        estimator.normalized_certainties_mutual_information()).numpy()
    print(model_name + " SE: ", ece_se, " MI: ", ece_mi)