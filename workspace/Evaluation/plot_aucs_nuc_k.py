import json
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

tfd = tfp.distributions
x = [5, 10, 25, 50, 100]
valid = False

with open('results_auroc_aupr.json') as json_file:
    data = json.load(json_file)

# AUROCs for nuc trained on validation data
c10_100_val_roc = data["nuc_val"]["CNN_cifar10_100"]["auroc"]
c10_1000_val_roc = data["nuc_val"]["CNN_cifar10_1000"]["auroc"]
c10_10000_val_roc = data["nuc_val"]["CNN_cifar10_10000"]["auroc"]
c10_val_roc = data["nuc_val"]["CNN_cifar10"]["auroc"]
c100_val_roc = data["nuc_val"]["CNN_cifar10_100"]["aupr"]
effnet_val_roc = data["nuc_val"]["effnetb3"]["auroc"]

# AUPRs for nuc trained on validation data
c10_100_val_pr = data["nuc_val"]["CNN_cifar10_100"]["aupr"]
c10_1000_val_pr = data["nuc_val"]["CNN_cifar10_1000"]["aupr"]
c10_10000_val_pr = data["nuc_val"]["CNN_cifar10_10000"]["aupr"]
c10_val_pr = data["nuc_val"]["CNN_cifar10"]["aupr"]
c100_val_pr = data["nuc_val"]["CNN_cifar100"]["aupr"]
effnet_val_pr = data["nuc_val"]["effnetb3"]["aupr"]

# AUROCs for nuc trained on train data
c10_100_tra_roc = data["nuc_train"]["CNN_cifar10_100"]["auroc"]
c10_1000_tra_roc = data["nuc_train"]["CNN_cifar10_1000"]["auroc"]
c10_10000_tra_roc = data["nuc_train"]["CNN_cifar10_10000"]["auroc"]
c10_tra_roc = data["nuc_train"]["CNN_cifar10"]["auroc"]
c100_tra_roc = data["nuc_train"]["CNN_cifar100"]["auroc"]
effnet_tra_roc = data["nuc_train"]["effnetb3"]["auroc"]

# AUPRs for nuc trained on train data
c10_100_tra_pr = data["nuc_train"]["CNN_cifar10_100"]["aupr"]
c10_1000_tra_pr = data["nuc_train"]["CNN_cifar10_1000"]["aupr"]
c10_10000_tra_pr = data["nuc_train"]["CNN_cifar10_10000"]["aupr"]
c10_tra_pr = data["nuc_train"]["CNN_cifar10"]["aupr"]
c100_tra_pr = data["nuc_train"]["CNN_cifar100"]["aupr"]
effnet_tra_pr = data["nuc_train"]["effnetb3"]["aupr"]


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_curves(auroc, valid):
    plt.xlabel("Anzahl Nachbarn (k)")
    plt.xticks([5, 10, 25, 50, 100])

    if auroc:
        plt.ylabel("AUROC")
        if valid:
            aucs = [c10_100_val_roc, c10_1000_val_roc, c10_10000_val_roc, c10_val_roc, c100_val_roc, effnet_val_roc]
        else:
            aucs = [c10_100_tra_roc, c10_1000_tra_roc, c10_10000_tra_roc, c10_tra_roc, c100_tra_roc, effnet_tra_roc]

    else:
        plt.ylabel("AUPR")
        if valid:
            aucs = [c10_100_val_pr, c10_1000_val_pr, c10_10000_val_pr, c10_val_pr, c100_val_pr, effnet_val_pr]
        else:
            aucs = [c10_100_tra_pr, c10_1000_tra_pr, c10_10000_tra_pr, c10_tra_pr, c100_tra_pr, effnet_tra_pr]

    means = []
    stdevs = []
    for values in aucs:
        means.append(tf.reduce_mean(values, axis=-1))
        stdevs.append(tf.math.reduce_std(values, axis=-1))

    colors = [adjust_lightness('b', 1.6), adjust_lightness('b', 1.3), adjust_lightness('b', 0.8),
              adjust_lightness('b', 0.4), 'tomato', 'yellowgreen']
    lbls = ["Cifar10 (100 Bilder)", "Cifar10 (1000 Bilder)", "Cifar10 (10000 Bilder)", "Cifar10", "Cifar100", "Cars196"]

    for mean, stdev, c, label in zip(means, stdevs, colors, lbls):
        plt.plot(x, mean, label=label, marker='.', color=c)
        plt.fill_between(x, mean + stdev, mean, color=c, alpha=0.15)
        plt.fill_between(x, mean, mean - stdev, color=c, alpha=0.15)


plt.figure(figsize=(9, 3.5))

plt.subplot(1, 2, 1)
plot_curves(True, valid)
box = plt.subplot(1, 2, 1).get_position()

plt.subplot(1, 2, 2)
plot_curves(False, valid)
box = plt.subplot(1, 2, 2).get_position()

plt.subplots_adjust(left=0.09, right=0.75, bottom=0.16, top=0.9, wspace=0.28, hspace=0.35)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot_name = '../plots/AUCs_nuc_different_k_valid.pdf' if valid else '../plots/AUCs_nuc_different_k_train.pdf'
plt.savefig(plot_name)
plt.show()