from uncertainty.Ensemble import ENSEMBLE_LOCATION, BaggingEns, DataAugmentationEns
from functions import get_train_and_test_data
import tensorflow as tf

NUM_MEMBERS = 5
MODEL = "CNN_cifar10"
METHOD = "data_augmentation"
DATA = "cifar10"

path_to_ensemble = ENSEMBLE_LOCATION + "/" + METHOD + "/" + MODEL
X_train, y_train, X_test, y_test, classes = get_train_and_test_data(DATA)

if METHOD == "bagging":
    estimator = BaggingEns(
        X_train, y_train, X_test, classes,
        model_name=MODEL, path_to_ensemble=path_to_ensemble, num_members=NUM_MEMBERS)
elif METHOD == "data_augmentation":
    estimator = DataAugmentationEns(
        X_train, y_train, X_test, classes,
        model_name=MODEL, path_to_ensemble=path_to_ensemble, num_members=NUM_MEMBERS
    )
else:
    raise NotImplementedError

pred = estimator.get_ensemble_prediction()


print("Simple model accuracy:" + str(estimator.get_simple_model_accuracy(y_test)))
print("Ensemble accuracy: " + str(estimator.get_ensemble_accuracy(y_test)) + "\n")
y_test = tf.argmax(y_test, axis=-1).numpy()

print("UNCERTAINTY BY SHANNON ENTROPY")
certainties = estimator.bounded_certainties_shannon_entropy()
index = sorted(range(len(certainties)), key=certainties.__getitem__, reverse=False)
preds = [pred[i] for i in index]
lbls = [y_test[i] for i in index]
certs = [certainties[i] for i in index]
print("targets:     " + str(lbls[:10]) + " ... " + str(lbls[-10:]))
print("predictions: " + str(preds[:10]) + " ... " + str(preds[-10:]))
print("certainties: " + str(certs[:10]) + " ... " + str(certs[-10:]) + "\n")


print("UNCERTAINTY BY MUTUAL INFORMATION")
certainties = estimator.bounded_certainties_mutual_information()
index = sorted(range(len(certainties)), key=certainties.__getitem__, reverse=False)
certs = [certainties[i] for i in index]
preds = [pred[i] for i in index]
lbls = [y_test[i] for i in index]
print("targets:     " + str(lbls[:10]) + " ... " + str(lbls[-10:]))
print("predictions: " + str(preds[:10]) + " ... " + str(preds[-10:]))
print("certainties: " + str(certs[:10]) + " ... " + str(certs[-10:]) + "\n")

scores = estimator.certainty_scores(y_test)
print("score SE: " + str(scores[0]) + "     score MI: " + str(scores[1]))

estimator.plot_diagrams(y_test)
