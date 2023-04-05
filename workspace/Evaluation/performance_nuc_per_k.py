import json
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from functions import adjust_lightness

tfd = tfp.distributions
x = [3, 5, 10, 25, 50, 100]

lbls = ["CNN Cifar10 (100 Bilder)", "CNN Cifar10 (1000 Bilder)", "CNN Cifar10 (10000 Bilder)",
        "CNN Cifar10 (gesamt)", "CNN Cifar100", "EfficientNet-B3 Cars"]

with open('../Results/auroc_aupr.json') as json_file:
    data = json.load(json_file)

# AUROCs for nuc trained on validation data
c10_1000_val_roc = data["NUC Va"]["CNN_cifar10_1000"]["auroc"]
c10_10000_val_roc = data["NUC Va"]["CNN_cifar10_10000"]["auroc"]
c10_val_roc = data["NUC Va"]["CNN_cifar10"]["auroc"]
c100_val_roc = data["NUC Va"]["CNN_cifar100"]["aupr"]
effnet_val_roc = data["NUC Va"]["effnetb3"]["auroc"]

# AUPRs for nuc trained on validation data
c10_1000_val_pr = data["NUC Va"]["CNN_cifar10_1000"]["aupr"]
c10_10000_val_pr = data["NUC Va"]["CNN_cifar10_10000"]["aupr"]
c10_val_pr = data["NUC Va"]["CNN_cifar10"]["aupr"]
c100_val_pr = data["NUC Va"]["CNN_cifar100"]["aupr"]
effnet_val_pr = data["NUC Va"]["effnetb3"]["aupr"]

# AUROCs for nuc trained on train data
c10_100_tra_roc = data["NUC Tr"]["CNN_cifar10_100"]["auroc"]
c10_1000_tra_roc = data["NUC Tr"]["CNN_cifar10_1000"]["auroc"]
c10_10000_tra_roc = data["NUC Tr"]["CNN_cifar10_10000"]["auroc"]
c10_tra_roc = data["NUC Tr"]["CNN_cifar10"]["auroc"]
c100_tra_roc = data["NUC Tr"]["CNN_cifar100"]["auroc"]
effnet_tra_roc = data["NUC Tr"]["effnetb3"]["auroc"]

# AUPRs for nuc trained on train data
c10_100_tra_pr = data["NUC Tr"]["CNN_cifar10_100"]["aupr"]
c10_1000_tra_pr = data["NUC Tr"]["CNN_cifar10_1000"]["aupr"]
c10_10000_tra_pr = data["NUC Tr"]["CNN_cifar10_10000"]["aupr"]
c10_tra_pr = data["NUC Tr"]["CNN_cifar10"]["aupr"]
c100_tra_pr = data["NUC Tr"]["CNN_cifar100"]["aupr"]
effnet_tra_pr = data["NUC Tr"]["effnetb3"]["aupr"]


def plot_eces():
    with open('../Results/eces.json') as json_file:
        eces = json.load(json_file)

    colors = [adjust_lightness('b', 1.8), adjust_lightness('b', 1.6), adjust_lightness('b', 1.0),
              adjust_lightness('b', 0.4), 'tomato', 'yellowgreen']
    models = ["CNN_cifar10_100", "CNN_cifar10_1000", "CNN_cifar10_10000", "CNN_cifar10", "CNN_cifar100", "effnetb3"]
    plt.figure(figsize=(9, 3))

    for count in range(2):
        ax = plt.subplot(1, 2, count+1)
        ax.set_axisbelow(True)
        plt.ylim(-0.01, 0.41)
        plt.grid(visible=True, color="gainsboro", linestyle='dashed', zorder=0)
        plt.xlabel("Anzahl Nachbarn (k)")
        plt.ylabel("ECE")
        plt.title("NUC Trainingsdaten" if count else "NUC Validierungsdaten")

        for (model, lbl, c) in zip(models, lbls, colors):
            try:
                values = eces[model]["NUC Tr" if count else "NUC Va"]
            except:
                continue
            means = tf.reduce_mean(values, axis=-1)
            stddevs = tf.math.reduce_std(values, axis=-1)
            plt.plot(x, means, label=lbl, marker='.', color=c, linestyle='dashed' if count else 'solid', zorder=1)
            plt.fill_between(x, means + stddevs, means - stddevs, color=c, alpha=0.1)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(left=0.07, right=0.72, bottom=0.16, top=0.91, wspace=0.25, hspace=0.35)
    plt.savefig('../Plots/ECEs_nuc_different_k.pdf')
    plt.show()


def plot_times(method):
    with open('../Results/times.json') as json_file:
        times = json.load(json_file)

    colors = [adjust_lightness('b', 1.8), adjust_lightness('b', 1.6), adjust_lightness('b', 1),
              adjust_lightness('b', 0.4), 'tomato', 'yellowgreen']
    models = ["CNN_cifar10_100", "CNN_cifar10_1000", "CNN_cifar10_10000", "CNN_cifar10", "CNN_cifar100", "effnetb3"]
    plt.figure(figsize=(9, 3))

    for key in ["preparation & uncertainty", "uncertainty"]:
        ax = plt.subplot(1, 2, 1 if key != "uncertainty" else 2)
        ax.set_axisbelow(True)
        plt.yscale("log")
        plt.ylim(9, 600) if key == "uncertainty" else plt.ylim(0.1, 600)
        plt.grid(visible=True, color="gainsboro", linestyle='dashed', zorder=0)
        plt.xlabel("Anzahl Nachbarn (k)")
        plt.ylabel("Laufzeit in Sekunden")
        plt.title("Training externes Netzwerk" if key != "uncertainty" else "Certainty Estimation")

        for (model, lbl, c) in zip(models, lbls, colors):
            if key != "uncertainty":
                means = tf.reduce_mean(times[model][method][key], axis=-1) - tf.reduce_mean(times[model][method]
                                                                                            ["uncertainty"], axis=-1)
                try:
                    means = tf.reduce_max([means, [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]], axis=0)
                except:
                    continue
                stddevs = tf.math.reduce_std(times[model][method][key], axis=-1) - tf.math.reduce_std(times[model][method]
                                                                                            ["uncertainty"], axis=-1)
            else:
                means = tf.reduce_mean(times[model][method][key], axis=-1)
                stddevs = tf.math.reduce_std(times[model][method][key], axis=-1)

            try:
                plt.plot(x, means, label=lbl, marker='.', color=c, linestyle='dashed' if method == "NUC Tr" else 'solid',
                         zorder=1)
            except:
                continue
            plt.fill_between(x, means + stddevs, means - stddevs, color=c, alpha=0.1)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(left=0.08, right=0.72, bottom=0.16, top=0.91, wspace=0.25, hspace=0.35)
    plt.savefig('../Plots/times_' + method + '_different_k.pdf')
    plt.show()


def plot_curves(auroc, valid):
    plt.xlabel("Anzahl Nachbarn (k)")
    plt.grid(visible=True, color="gainsboro", linestyle='dashed', zorder=0)

    if auroc:
        plt.ylabel("AUROC")
        if valid:
            aucs = [[0.], c10_1000_val_roc, c10_10000_val_roc, c10_val_roc, c100_val_roc, effnet_val_roc]
        else:
            aucs = [c10_100_tra_roc, c10_1000_tra_roc, c10_10000_tra_roc, c10_tra_roc, c100_tra_roc, effnet_tra_roc]

    else:
        plt.ylabel("AUPR")
        if valid:
            aucs = [[0.], c10_1000_val_pr, c10_10000_val_pr, c10_val_pr, c100_val_pr, effnet_val_pr]
        else:
            aucs = [c10_100_tra_pr, c10_1000_tra_pr, c10_10000_tra_pr, c10_tra_pr, c100_tra_pr, effnet_tra_pr]

    means = []
    stdevs = []
    for values in aucs:
        means.append(tf.reduce_mean(values, axis=-1))
        stdevs.append(tf.math.reduce_std(values, axis=-1))

    colors = [adjust_lightness('b', 1.8), adjust_lightness('b', 1.6), adjust_lightness('b', 1.0),
              adjust_lightness('b', 0.4), 'tomato', 'yellowgreen']

    for mean, stdev, c, label in zip(means, stdevs, colors, lbls):
        if valid and label == "CNN Cifar10 (100 Bilder)":
            continue
        plt.plot(x, mean, label=label, marker='.', color=c, linestyle='dashed' if not valid else 'solid', zorder=1)
        plt.fill_between(x, mean + stdev, mean - stdev, color=c, alpha=0.1)


plt.figure(figsize=(9, 3))

ax = plt.subplot(1, 2, 1)
ax.set_axisbelow(True)
plot_curves(True, True)

ax = plt.subplot(1, 2, 2)
plot_curves(False, True)
ax.set_axisbelow(True)

plt.subplots_adjust(left=0.07, right=0.72, bottom=0.16, top=0.96, wspace=0.25, hspace=0.35)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot_name = '../Plots/AUCs_nuc_different_k_valid.pdf'
plt.savefig(plot_name)
plt.show()

plt.figure(figsize=(9, 3))

ax = plt.subplot(1, 2, 1)
ax.set_axisbelow(True)
plot_curves(True, False)

ax = plt.subplot(1, 2, 2)
plot_curves(False, False)
ax.set_axisbelow(True)

plt.subplots_adjust(left=0.07, right=0.72, bottom=0.16, top=0.96, wspace=0.25, hspace=0.35)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot_name = '../Plots/AUCs_nuc_different_k_train.pdf'
plt.savefig(plot_name)
plt.show()


plot_times("NUC Tr")
plot_times("NUC Va")

plot_eces()
