from functions import CNN, get_train_and_test_data
import tensorflow as tf
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier

model_name = "CNN_cifar100"
data = "cifar100"
checkpoint_path = "../models/classification/" + model_name + "/cp.ckpt"
path_uncertainty_model = "../models/classification/uncertainty_model/" + data + "/cp.ckpt"

model = CNN(classes=100 if model_name == "CNN_cifar100" else 10)
model.load_weights(checkpoint_path)
xtrain, ytrain, xtest, ytest, _ = get_train_and_test_data(data)

# uncomment for uncertainty estimation on data in the train data set that has not been used for training
#xtest = xtrain[1000:]
#ytest = ytrain[1000:]

model.evaluate(xtest, ytest)
ytrain = tf.argmax(ytrain, axis=-1).numpy()
ytest = tf.argmax(ytest, axis=-1).numpy()

estimator = NeighborhoodUncertaintyClassifier(model, xtrain, ytrain, xtest, ytest,
                                              path_uncertainty_model=path_uncertainty_model)

certainties = estimator.certainties
index = sorted(range(len(certainties)), key=certainties.__getitem__, reverse=False)
print("most and least uncertain samples: " + str(index[0]) + ": " + str(certainties[index[0]]) + ",   " +
      str(index[-1]) + ": " + str(certainties[index[-1]]))
ypred = tf.argmax(model.predict(xtest), axis=-1).numpy()
preds = [ypred[i] for i in index]
lbls = [ytest[i] for i in index]
certs = [certainties[i] for i in index]
print("targets:     " + str(lbls[:10]) + " ... " + str(lbls[-10:]))
print("predictions: " + str(preds[:10]) + " ... " + str(preds[-10:]))
print("certainties: " + str(certs[:10]) + " ... " + str(certs[-10:]))

estimator.plot_diagrams()
print("score = ", estimator.certainty_score())

'''
corrects = []
for i in range(int(len(lbls)/10)):
    correct = [l == p for l, p in zip(lbls[i*10:(i+1)*10], preds[i*10:(i+1)*10])]
    corrects.append(list(correct).count(True))

plt.title("Neighborhood Uncertainty")
plt.xlabel("groups of increasing certainty")
plt.ylabel("number of correct classifications")
plt.ylim(-0.05, 10.05)
plt.scatter([i for i in range(len(corrects))], corrects)
plt.show()
'''
