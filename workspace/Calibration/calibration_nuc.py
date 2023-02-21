import matplotlib.pyplot as plt
import tensorflow as tf
import sys
sys.path.append("/home/urz/hlichten")
from uncertainty.NeighborhoodUncertainty import NeighborhoodUncertaintyClassifier
from uncertainty.calibration_classification import reliability_diagram, expected_calibration_error
from functions import get_data, CNN, split_validation_from_train, build_effnet, COLORS

validation = True
fig = plt.figure(figsize=(11, 3))
method_name = "NUC Va" if validation else "NUC Tr"

for count, model_name in enumerate(["CNN_cifar10_1000", "CNN_cifar10", "effnetb3"]):
    xtrain, ytrain, xval, yval, xtest, ytest, classes = get_data("cifar10" if count != 2 else "cars196",
                                                                       num_data=1000 if count == 0 else None)
    model = tf.keras.models.load_model("../models/classification/" + model_name)
    ypred = model.predict(xtest)
    ax = plt.subplot(1, 3, count + 1)
    ax.set_axisbelow(True)
    plt.grid(visible=True, color="gainsboro", linestyle='dashed', zorder=0)

    if validation:
        path = "../models/classification/uncertainty_model/val/10/"
        if count != 2:
            xtrain, ytrain = xval[:int(4 * len(xval) / 5)], yval[:int(4 * len(yval) / 5)]
            xval, yval = xval[int(4 * len(xval) / 5):], yval[int(4 * len(yval) / 5):]
        else:
            xtrain, ytrain, xval, yval = split_validation_from_train(xval, yval, classes, num_imgs_per_class=2)
    else:
        path = "../models/classification/uncertainty_model/train/10/"

    estimator = NeighborhoodUncertaintyClassifier(model, xtrain, ytrain, xval, yval, xtest,
                                                  path + model_name + "/cp.ckpt", k=10)

    reliability_diagram(y_true=tf.argmax(ytest, axis=-1), output=ypred, certainties=estimator.certainties,
                        label_perfectly_calibrated=count == 2, num_bins=15, color=COLORS[method_name],
                        method="Testprädiktionen" if count == 2 else None)
    ece = expected_calibration_error(tf.argmax(ytest, axis=-1), tf.argmax(ypred, axis=-1), estimator.certainties).numpy()
    print(ece)
    plt.text(0.02, 0.95, "ECE: {:.3f}".format(ece), color="brown", weight="bold")

    if count == 2:
        plt.legend(bbox_to_anchor=(1.02, 1))
        plt.title("EfficientNet-B3")
    elif count == 1:
        plt.title("CNN Cifar10 (gesamt)")
    else:
        plt.title("CNN Cifar10 (1000 Bilder)")


plt.subplots_adjust(left=0.06, right=0.82, bottom=0.16, top=0.9, wspace=0.3, hspace=0.35)
plot_name = '../plots/calibration_nuc_on_validation.png' if validation else '../plots/calibration_nuc_on_train.png'

plt.savefig(plot_name, dpi=300)
plt.show()