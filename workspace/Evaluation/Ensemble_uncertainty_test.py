from uncertainty.Ensemble import ENSEMBLE_LOCATION, BaggingEns, DataAugmentationEns
from functions import get_train_and_test_data

PATH_TO_PRETRAINED_RESNET_10 = "../models/classification/ResNet_cifar10_1000/cp.ckpt"
NUM_MEMBERS = 5
MODEL = "ResNet_cifar10"
METHOD = "bagging"
DATA = "1000"

path_to_ensemble = ENSEMBLE_LOCATION + "/" + METHOD + "/" + MODEL + "_" + DATA
X_train, y_train, X_test, y_test, classes = get_train_and_test_data("cifar10")

if METHOD == "bagging":
    estimator = BaggingEns(
        X_train, y_train, X_test, classes,
        model_name=MODEL, path_to_ensemble=path_to_ensemble, num_members=NUM_MEMBERS)
#elif METHOD == "dataset_partition":
 #   estimator = PartitionEns(
  #      X_train, y_train, X_test, classes,
   #     model_name=MODEL, path_to_ensemble=path_to_ensemble, num_members=NUM_MEMBERS
    #)
elif METHOD == "data_augmentation":
    estimator = DataAugmentationEns(
        X_train, y_train, X_test, classes,
        model_name=MODEL, path_to_ensemble=path_to_ensemble, num_members=NUM_MEMBERS
    )
else:
    raise NotImplementedError

pred = estimator.get_ensemble_prediction()

print(MODEL.upper() + " - ENSEMBLE")
print("Params: data=" + DATA + " method=" + METHOD.upper() + " #members=" + str(NUM_MEMBERS))
print("Simple model accuracy:" + str(estimator.get_simple_model_accuracy(y_test)))
print("Ensemble accuracy: " + str(estimator.get_ensemble_accuracy(y_test)) + "\n")

print("UNCERTAINTY BY MEAN (Shannon Entropy)")
certainties = estimator.get_certainties_by_SE()
index = sorted(range(len(certainties)), key=certainties.__getitem__, reverse=False)
preds = [pred[i] for i in index]
lbls = [y_test[i] for i in index]
certs = [certainties[i] for i in index]
print("targets:     " + str(lbls[:10]) + " ... " + str(lbls[-10:]))
print("predictions: " + str(preds[:10]) + " ... " + str(preds[-10:]))
print("certainties: " + str(certs[:10]) + " ... " + str(certs[-10:]) + "\n")


print("UNCERTAINTY BY STANDARD DEVIATION")
certainties = estimator.get_certainties_by_stddev()
index = sorted(range(len(certainties)), key=certainties.__getitem__, reverse=False)
certs = [certainties[i] for i in index]
preds = [pred[i] for i in index]
lbls = [y_test[i] for i in index]
print("targets:     " + str(lbls[:10]) + " ... " + str(lbls[-10:]))
print("predictions: " + str(preds[:10]) + " ... " + str(preds[-10:]))
print("certainties: " + str(certs[:10]) + " ... " + str(certs[-10:]) + "\n")

scores = estimator.certainty_scores(y_test)
print("score SE: " + str(scores[0]) + "     score StdDev: " + str(scores[1]))

estimator.plot_diagrams(y_test)
