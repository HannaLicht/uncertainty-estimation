# use mnist & fashion_mnist mit random Farben, noise, malaria, rock-paper-scissors
import matplotlib.pyplot as plt

from functions import get_train_and_test_data, split_validation_from_train, build_effnet, CNN
from uncertainty.Ensemble import DataAugmentationEns, RandomInitShuffleEns, BaggingEns, ENSEMBLE_LOCATION
from uncertainty.MC_Dropout import MCDropoutEstimator
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
from uncertainty.calibration_classification import get_normalized_certainties
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

model_name = "CNN_cifar10_100"
data = "cifar10"

path_to_bagging_ens = ENSEMBLE_LOCATION + "/bagging/" + model_name
path_to_dataAug_ens = ENSEMBLE_LOCATION + "/data_augmentation/" + model_name
path_to_randInitShuffle_ens = ENSEMBLE_LOCATION + "/rand_initialization_shuffle/" + model_name
model_path = "../models/classification/" + model_name + "/cp.ckpt"
path_to_nuc_val = "../models/classification/uncertainty_model/" + model_name + "/cp.ckpt"
path_to_nuc_train = "../models/classification/uncertainty_model/trained_on_traindata/" + model_name + "/cp.ckpt"

# TODO get data

xtest, ytest = [], []

# TODO: get train and val data of original dataset nuc is trained on
xtrain, ytrain, xval, yval, _, _, num_classes = get_train_and_test_data(data, validation_test_split=True)

model = build_effnet(num_classes) if model_name == "effnetb3" else CNN(classes=num_classes)
model.load_weights(model_path)

if model_name != "effnetb3":
    xtrain_nuc_val, ytrain_nuc_val = xval[:int(4 * len(xval) / 5)], yval[:int(4 * len(yval) / 5)]
    xval_nuc_val, yval_nuc_val = xval[int(4 * len(xval) / 5):], yval[int(4 * len(yval) / 5):]
else:
    xtrain_nuc_val, ytrain_nuc_val, xval_nuc_val, yval_nuc_val = \
        split_validation_from_train(xval, yval, num_classes, num_imgs_per_class=2)

MCEstimator = MCDropoutEstimator(model, xtest, num_classes, T=50, xval=xval, yval=yval)
DAEstimator = DataAugmentationEns(xtest, num_classes, path_to_dataAug_ens, X_val=xval, y_val=yval, val=True)
RISEstimator = RandomInitShuffleEns(xtest, num_classes, path_to_randInitShuffle_ens, X_val=xval, y_val=yval, val=True)
BaEstimator = BaggingEns(xtest, num_classes, path_to_bagging_ens, X_val=xval, y_val=yval, val=True)
nuc_train = NeighborhoodUncertaintyClassifier(model, xtrain, ytrain, xval, yval, xtest, path_to_nuc_train)
nuc_val = NeighborhoodUncertaintyClassifier(model, xtrain_nuc_val, ytrain_nuc_val, xval_nuc_val, yval_nuc_val,
                                            xtest, path_to_nuc_val)

soft_ent_uncert_test = tfd.Categorical(probs=model.predict(xtest, verbose=0)).entropy().numpy()
soft_ent_uncert_val = tfd.Categorical(probs=model.predict(xval, verbose=0)).entropy().numpy()
softmax_entropy = get_normalized_certainties(model.predict(xval, verbose=0), yval,
                                             soft_ent_uncert_val, soft_ent_uncert_test)
mcdr_se = MCEstimator.normalized_certainties_shannon_entropy()
mcdr_mi = MCEstimator.normalized_certainties_mutual_information()
bag_se = BaEstimator.normalized_certainties_shannon_entropy()
bag_mi = BaEstimator.normalized_certainties_mutual_information()
rand_se = RISEstimator.normalized_certainties_shannon_entropy()
rand_mi = RISEstimator.normalized_certainties_mutual_information()
aug_se = DAEstimator.normalized_certainties_shannon_entropy()
aug_mi = DAEstimator.normalized_certainties_mutual_information()

certainties = [mcdr_se, mcdr_mi,
               bag_se, bag_mi, rand_se, rand_mi, aug_se, aug_mi,
               nuc_train.certainties, nuc_val.certainties, softmax_entropy]
methods = ["MCdrop SE", "MCdrop MI",
           "Bag SE", "Bag MI", "Rand SE", "Rand MI", "DataAug SE", "DataAug MI",
           "NUC_train", "NUC_valid", "Softmax"]
thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

plt.xlabel("Schwellenwerte für die Konfidenz")
plt.ylabel("Recall")
plt.figure(figsize=(15, 15))

for certs, method in zip(certainties, methods):
    TU = [certs.count(lambda x: x < thr) for thr in thresholds]
    FC = [certs.count(lambda x: x >= thr) for thr in thresholds]
    recall = tf.divide(TU, tf.add(TU, FC))
    plt.plot(thresholds, recall, label=method)

plt.legend(loc="lower right")
plt.show()

