import keras.applications.efficientnet as efn
from functions import create_simple_model, get_test_data, ResNet
from uncertainty.MC_Dropout import MCDropoutEstimator

T = 50
MODEL = "ResNet_cifar10_1000"
CHECKPOINT_PATH = "../models/classification/" + MODEL + "/cp.ckpt"

if MODEL == "simple_seq_model_mnist":
    x, y, num_classes = get_test_data("mnist")
    model = create_simple_model()
    model.load_weights(CHECKPOINT_PATH)
    estimator = MCDropoutEstimator(model, x, num_classes, T)
elif MODEL == "simple_seq_model_fashion_mnist":
    x, y, num_classes = get_test_data("fashion_mnist")
    model = create_simple_model()
    model.load_weights(CHECKPOINT_PATH)
    estimator = MCDropoutEstimator(model, x, num_classes, T)
elif MODEL == "ResNet_cifar100":
    x, y, num_classes = get_test_data("cifar100")
    model = ResNet(classes=100)
    model.load_weights(CHECKPOINT_PATH)
    estimator = MCDropoutEstimator(model, x, num_classes, T)
elif MODEL == "ResNet_cifar10_1000":
    x, y, num_classes = get_test_data("cifar10")
    model = ResNet(classes=10)
    model.load_weights(CHECKPOINT_PATH)
    estimator = MCDropoutEstimator(model, x, num_classes, T)
elif MODEL == "effnetb0":
    x, y, num_classes = get_test_data("imagenette")
    eff = efn.EfficientNetB0(weights='imagenet')
    estimator = MCDropoutEstimator(eff, x, num_classes, T)
else:
    raise NotImplementedError

pred = estimator.get_ensemble_prediction()

print(MODEL.upper() + " - FULL MC DROPOUT \nParams: #T=" + str(T))
print("Simple model accuracy:" + str(estimator.get_simple_model_accuracy(y)))
print("Ensemble accuracy: " + str(estimator.get_ensemble_accuracy(y)) + "\n")

print("UNCERTAINTY BY SE")
certainties = estimator.get_certainties_by_SE()
index = sorted(range(len(certainties)), key=certainties.__getitem__, reverse=False)
preds = [pred[i] for i in index]
lbls = [y[i] for i in index]
certs = [certainties[i] for i in index]
print("targets:     " + str(lbls[:10]) + " ... " + str(lbls[-10:]))
print("predictions: " + str(preds[:10]) + " ... " + str(preds[-10:]))
print("certainties: " + str(certs[:10]) + " ... " + str(certs[-10:]) + "\n")


print("UNCERTAINTY BY MI")
certainties = estimator.get_certainties_by_mutual_inf()
index = sorted(range(len(certainties)), key=certainties.__getitem__, reverse=False)
certs = [certainties[i] for i in index]
preds = [pred[i] for i in index]
lbls = [y[i] for i in index]
print("targets:     " + str(lbls[:10]) + " ... " + str(lbls[-10:]))
print("predictions: " + str(preds[:10]) + " ... " + str(preds[-10:]))
print("certainties: " + str(certs[:10]) + " ... " + str(certs[-10:]) + "\n")

scores = estimator.certainty_scores(y)
print("score SE: " + str(scores[0]) + "     score MI: " + str(scores[1]))

estimator.plot_diagrams(y)

'''
corrects = []
for i in range(int(len(lbls)/10)):
    correct = [l == p for l, p in zip(lbls[i*10:(i+1)*10], preds[i*10:(i+1)*10])]
    corrects.append(list(correct).count(True))

plt.title("MC DRopout - Stddev Uncertainty")
plt.xlabel("groups of increasing certainty")
plt.ylabel("number of correct classifications")
plt.ylim(-0.05, 10.05)
plt.scatter([i for i in range(len(corrects))], corrects)
plt.show()
'''