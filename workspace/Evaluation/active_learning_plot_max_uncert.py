import json
import matplotlib.pyplot as plt
import tensorflow as tf
from functions import COLORS

startdata = 100
num_images = 100
smaller_u = False

TIMES_IMAGES_ADDED = 10

end = "_smallU" if smaller_u else ""
with open('../Results/active_learning/' + str(startdata) + "_" + str(num_images) + end + ".json") as json_file:
    results = json.load(json_file)

numbers = [startdata + (i+1)*num_images for i in range(TIMES_IMAGES_ADDED)]
rand = [tf.reduce_mean(results[str(imgs)]["random"]) for imgs in numbers]
bag_pe = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["BaggingEns"]["PE"]) for imgs in numbers]
bag_mi = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["BaggingEns"]["MI"]) for imgs in numbers]
aug_pe = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["PE"]) for imgs in numbers]
aug_mi = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["MI"]) for imgs in numbers]

rand_var = [tf.math.reduce_std(results[str(imgs)]["random"]) for imgs in numbers]
bag_pe_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["BaggingEns"]["PE"]) for imgs in numbers]
bag_mi_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["BaggingEns"]["MI"]) for imgs in numbers]
aug_pe_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["PE"]) for imgs in numbers]
aug_mi_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["MI"]) for imgs in numbers]

softmax = [tf.reduce_mean(results[str(imgs)]["softmax_entropy"]) for imgs in numbers]
mc_pe = [tf.reduce_mean(results[str(imgs)]["MC_drop"]["PE"]) for imgs in numbers]
mc_mi = [tf.reduce_mean(results[str(imgs)]["MC_drop"]["MI"]) for imgs in numbers]
nuc = [tf.reduce_mean(results[str(imgs)]["NUC Va"]) for imgs in numbers]
nuc_tr = [tf.reduce_mean(results[str(imgs)]["NUC Tr"]) for imgs in numbers]

softmax_var = [tf.math.reduce_std(results[str(imgs)]["softmax_entropy"]) for imgs in numbers]
mc_pe_var = [tf.math.reduce_std(results[str(imgs)]["MC_drop"]["PE"]) for imgs in numbers]
mc_mi_var = [tf.math.reduce_std(results[str(imgs)]["MC_drop"]["MI"]) for imgs in numbers]
nuc_var = [tf.math.reduce_std(results[str(imgs)]["NUC Va"]) for imgs in numbers]
nuc_tr_var = [tf.math.reduce_std(results[str(imgs)]["NUC Tr"]) for imgs in numbers]

plt.figure(figsize=(11.5, 3.5))


def plot_accs(values, var, l):
    if l == "Zufall":
        c = "black"
        s = "--"
        alpha = 0.05
    elif l == "Bag PE" or l == "Bag MI" or l == "NUC Tr" or l == "MCD PE" or l == "MCD MI":
        c = COLORS[l]
        s = "-"
        alpha = 0.07
    else:
        c = COLORS[l]
        s = "-"
        alpha = 0.1
    plt.plot(numbers, values, s, label=l, color=c, zorder=1, linewidth=1.2)
    plt.fill_between(numbers, tf.math.add(values, var), tf.math.subtract(values, var), color=c, alpha=alpha, zorder=0)
    if l == "Zufall":
        plt.plot(numbers, values, label=" ", color="white", zorder=0)


def decore_plot(ax):
    plt.xlabel("Anzahl gelabelter Bilder")
    plt.ylabel("Testaccuracy")
    plt.grid(visible=True, color="gainsboro", linestyle='dashed', zorder=0)
    ax.set_axisbelow(True)
    plt.legend(loc="lower right")
    if STARTDATA == 100 and smaller_u:
        plt.ylim(0.405, 0.542)
    elif STARTDATA == 100:
        plt.ylim(0.379, 0.533)
    elif STARTDATA == 1000 and NUM_IMAGES == 100:
        plt.ylim(0.526, 0.581)
        plt.legend(loc="upper left")
    elif STARTDATA == 10000:
        plt.ylim(0.649, 0.709)
    else:
        plt.ylim(0.539, 0.683)


ax = plt.subplot(1, 3, 1)
means = [rand, bag_pe, bag_mi, aug_pe, aug_mi]
stddevs = [rand_var, bag_pe_var, bag_mi_var, aug_pe_var, aug_mi_var]
lbls = ["Zufall", "Bag PE", "Bag MI", "DA PE", "DA MI"]
for values, var, l in zip(means, stddevs, lbls):
    plot_accs(values, var, l)
decore_plot(ax)

ax = plt.subplot(1, 3, 2)
for values, var, l in zip([rand, mc_pe, mc_mi, softmax],
                       [rand_var, mc_pe_var, mc_mi_var, softmax_var],
                       ["Zufall", "MCD PE", "MCD MI", "SE"]):
    plot_accs(values, var, l)
decore_plot(ax)

ax = plt.subplot(1, 3, 3)
for values, var, l in zip([rand, nuc_tr, nuc],
                       [rand_var, nuc_tr_var, nuc_var],
                       ["Zufall", "NUC Tr", "NUC Va"]):
    plot_accs(values, var, l)
decore_plot(ax)

plt.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.95, wspace=0.25, hspace=0.35)
end = "_smallU" if smaller_u else ""
plt.savefig("../plots/active_learning_" + str(startdata) + "_" + str(num_images) + end + ".pdf")

plt.show()

# plot the best methods for different sizes of initial train data set and different batch sizes
plt.figure(figsize=(11.5, 3.7))
if startdata == 1000 and num_images == 1000:

    ax = plt.subplot(1, 3, 2)
    plt.title("Start: 1000  Hinzu: 1000")
    for values, var, l in zip([rand, aug_mi, mc_mi, nuc],
                              [rand_var, aug_mi_var, mc_mi_var, nuc_var],
                              ["Zufall", "DA MI", "MCD MI", "NUC Va"]):
        plot_accs(values, var, l)
    decore_plot(ax)

    NUM_IMAGES = 100

    with open('../Results/active_learning/' + str(startdata) + "_" + str(NUM_IMAGES) + ".json") as json_file:
        results = json.load(json_file)

    numbers = [startdata + (i + 1) * NUM_IMAGES for i in range(TIMES_IMAGES_ADDED)]
    rand = [tf.reduce_mean(results[str(imgs)]["random"]) for imgs in numbers]
    mc_mi = [tf.reduce_mean(results[str(imgs)]["MC_drop"]["MI"]) for imgs in numbers]
    nuc = [tf.reduce_mean(results[str(imgs)]["NUC Va"]) for imgs in numbers]

    mc_mi_var = [tf.math.reduce_std(results[str(imgs)]["MC_drop"]["MI"]) for imgs in numbers]
    nuc_var = [tf.math.reduce_std(results[str(imgs)]["NUC Va"]) for imgs in numbers]
    rand_var = [tf.math.reduce_std(results[str(imgs)]["random"]) for imgs in numbers]

    ax = plt.subplot(1, 3, 1)
    plt.title("Start: 1000  Hinzu: 100")
    for values, var, l in zip([rand, mc_mi, nuc],
                              [rand_var, mc_mi_var, nuc_var],
                              ["Zufall", "MCD MI", "NUC Va"]):
        plot_accs(values, var, l)
    decore_plot(ax)

    STARTDATA = 10000
    NUM_IMAGES = 1000

    with open('../Results/active_learning/' + str(STARTDATA) + "_" + str(NUM_IMAGES) + ".json") as json_file:
        results = json.load(json_file)

    numbers = [STARTDATA + (i + 1) * NUM_IMAGES for i in range(TIMES_IMAGES_ADDED)]
    rand = [tf.reduce_mean(results[str(imgs)]["random"]) for imgs in numbers]
    aug_mi = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["MI"]) for imgs in numbers]
    mc_mi = [tf.reduce_mean(results[str(imgs)]["MC_drop"]["MI"]) for imgs in numbers]
    bag_pe = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["BaggingEns"]["PE"]) for imgs in numbers]

    mc_mi_var = [tf.math.reduce_std(results[str(imgs)]["MC_drop"]["MI"]) for imgs in numbers]
    bag_pe_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["BaggingEns"]["PE"]) for imgs in numbers]
    rand_var = [tf.math.reduce_std(results[str(imgs)]["random"]) for imgs in numbers]
    aug_mi_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["MI"]) for imgs in numbers]

    ax = plt.subplot(1, 3, 3)
    plt.title("Start: 10000  Hinzu: 1000")
    for values, var, l in zip([rand, aug_mi, mc_mi, bag_pe],
                              [rand_var, aug_mi_var, mc_mi_var, bag_pe_var],
                              ["Zufall", "DA MI", "MCD MI", "Bag PE"]):
        plot_accs(values, var, l)
    decore_plot(ax)
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.93, wspace=0.25, hspace=0.35)
    plt.savefig("../plots/best_active_learning.pdf")
    plt.show()
