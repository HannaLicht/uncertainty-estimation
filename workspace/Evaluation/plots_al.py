import json
import matplotlib.pyplot as plt
import tensorflow as tf
from functions import COLORS

STARTDATA = 10000
NUM_IMAGES = 1000
HYBRID = True


if STARTDATA == 100:
    times_images_added = 10
elif STARTDATA == 10000 or (STARTDATA == 1000 and NUM_IMAGES == 100):
    times_images_added = 10
else:
    times_images_added = 9


with open('../Results/active_learning/' + str(STARTDATA) + "_" + str(NUM_IMAGES) + ".json") as json_file:
    results = json.load(json_file)

numbers = [STARTDATA + (i+1)*NUM_IMAGES for i in range(times_images_added)]
rand = [tf.reduce_mean(results[str(imgs)]["random"]) for imgs in numbers]
bag_se = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["BaggingEns"]["SE"]) for imgs in numbers]
bag_mi = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["BaggingEns"]["MI"]) for imgs in numbers]
aug_se = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["SE"]) for imgs in numbers]
aug_mi = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["MI"]) for imgs in numbers]

rand_var = [tf.math.reduce_std(results[str(imgs)]["random"]) for imgs in numbers]
bag_se_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["BaggingEns"]["SE"]) for imgs in numbers]
bag_mi_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["BaggingEns"]["MI"]) for imgs in numbers]
aug_se_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["SE"]) for imgs in numbers]
aug_mi_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["MI"]) for imgs in numbers]

if HYBRID:
    with open('../Results/active_learning/' + str(STARTDATA) + "_" + str(NUM_IMAGES) + "_hybrid.json") as json_file:
        results = json.load(json_file)
    just_div = [tf.reduce_mean(results[str(imgs)]["just_divers"]) for imgs in numbers]
    just_div_var = [tf.math.reduce_std(results[str(imgs)]["just_divers"]) for imgs in numbers]

softmax = [tf.reduce_mean(results[str(imgs)]["softmax_entropy"]) for imgs in numbers]
mc_se = [tf.reduce_mean(results[str(imgs)]["MC_drop"]["SE"]) for imgs in numbers]
mc_mi = [tf.reduce_mean(results[str(imgs)]["MC_drop"]["MI"]) for imgs in numbers]
nuc = [tf.reduce_mean(results[str(imgs)]["NUC"]) for imgs in numbers]
nuc_tr = [tf.reduce_mean(results[str(imgs)]["NUC Tr"]) for imgs in numbers]

softmax_var = [tf.math.reduce_std(results[str(imgs)]["softmax_entropy"]) for imgs in numbers]
mc_se_var = [tf.math.reduce_std(results[str(imgs)]["MC_drop"]["SE"]) for imgs in numbers]
mc_mi_var = [tf.math.reduce_std(results[str(imgs)]["MC_drop"]["MI"]) for imgs in numbers]
nuc_var = [tf.math.reduce_std(results[str(imgs)]["NUC"]) for imgs in numbers]
nuc_tr_var = [tf.math.reduce_std(results[str(imgs)]["NUC Tr"]) for imgs in numbers]


plt.figure(figsize=(11.5, 3.5))


def plot_accs(values, var, l):
    if l == "Zufall":
        c = "black"
        s = "--"
    elif l == "Diversität":
        c = "purple"
        s = ":"
    else:
        c = COLORS[l]
        s = "-"
    plt.plot(numbers, values, s, label=l, color=c, zorder=1, linewidth=1.2)
    plt.fill_between(numbers, tf.math.add(values, var), tf.math.subtract(values, var), color=c, alpha=0.08, zorder=0)
    if l == "Zufall":
        plt.plot(numbers, values, label=" ", color="white", zorder=0)


def decore_plot(ax):
    plt.xlabel("Anzahl gelabelter Bilder")
    plt.ylabel("Testaccuracy")
    plt.grid(visible=True, color="gainsboro", linestyle='dashed', zorder=0)
    ax.set_axisbelow(True)
    plt.legend(loc="lower right")
    if STARTDATA == 100:
        plt.ylim(0.38, 0.521)
    elif STARTDATA == 1000 and NUM_IMAGES == 100:
        plt.ylim(0.508, 0.565)
    elif STARTDATA == 10000:
        plt.ylim(0.642, 0.706)


ax = plt.subplot(1, 3, 1)
means = [rand, just_div] if HYBRID else [rand, bag_se, bag_mi, aug_se, aug_mi]
stddevs = [rand_var, just_div_var] if HYBRID else [rand_var, bag_se_var, bag_mi_var, aug_se_var, aug_mi_var]
lbls = ["Zufall", "Diversität"] if HYBRID else ["Zufall", "Bag SE", "Bag MI", "DA SE", "DA MI"]
for values, var, l in zip(means, stddevs, lbls):
    plot_accs(values, var, l)
decore_plot(ax)

ax = plt.subplot(1, 3, 2)
for values, var, l in zip([rand, mc_se, mc_mi, softmax],
                       [rand_var, mc_se_var, mc_mi_var, softmax_var],
                       ["Zufall", "MCD SE", "MCD MI", "Soft SE"]):
    plot_accs(values, var, l)
decore_plot(ax)

ax = plt.subplot(1, 3, 3)
for values, var, l in zip([rand, nuc_tr, nuc],
                       [rand_var, nuc_tr_var, nuc_var],
                       ["Zufall", "NUC Tr", "NUC Va"]):
    plot_accs(values, var, l)
decore_plot(ax)

plt.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.95, wspace=0.25, hspace=0.35)
div = "_hybrid" if HYBRID else ""
plt.savefig("../plots/active_learning_" + str(STARTDATA) + "_" + str(NUM_IMAGES) + div + ".pdf")

plt.show()

plt.figure(figsize=(10, 4.5))
if STARTDATA == 1000 and NUM_IMAGES == 1000 and not HYBRID:

    ax = plt.subplot(1, 2, 1)
    plt.title("Start mit 1000 Bildern")
    for values, var, l in zip([rand, bag_mi, mc_mi, nuc],
                              [rand_var, bag_mi_var, mc_mi_var, nuc_var],
                              ["Zufall", "Bag MI", "MCD MI", "NUC Va"]):
        plot_accs(values, var, l)
    decore_plot(ax)

    times_images_added = 10
    STARTDATA = 10000

    with open('../Results/active_learning/' + str(STARTDATA) + "_" + str(NUM_IMAGES) + ".json") as json_file:
        results = json.load(json_file)

    numbers = [STARTDATA + (i + 1) * NUM_IMAGES for i in range(times_images_added)]
    rand = [tf.reduce_mean(results[str(imgs)]["random"]) for imgs in numbers]
    bag_se = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["BaggingEns"]["SE"]) for imgs in numbers]
    bag_mi = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["BaggingEns"]["MI"]) for imgs in numbers]
    aug_mi = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["MI"]) for imgs in numbers]
    mc_mi = [tf.reduce_mean(results[str(imgs)]["MC_drop"]["MI"]) for imgs in numbers]
    nuc = [tf.reduce_mean(results[str(imgs)]["NUC"]) for imgs in numbers]

    mc_mi_var = [tf.math.reduce_std(results[str(imgs)]["MC_drop"]["MI"]) for imgs in numbers]
    nuc_var = [tf.math.reduce_std(results[str(imgs)]["NUC"]) for imgs in numbers]
    rand_var = [tf.math.reduce_std(results[str(imgs)]["random"]) for imgs in numbers]
    bag_se_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["BaggingEns"]["SE"]) for imgs in numbers]
    bag_mi_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["BaggingEns"]["MI"]) for imgs in numbers]
    aug_mi_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["MI"]) for imgs in numbers]

    ax = plt.subplot(1, 2, 2)
    plt.title("Start mit 10000 Bildern")
    for values, var, l in zip([rand,  bag_se, bag_mi, aug_mi, mc_mi, nuc],
                              [rand_var, bag_se_var, bag_mi_var, aug_mi_var, mc_mi_var, nuc_var],
                              ["Zufall", "Bag SE", "Bag MI", "DA MI", "MCD MI", "NUC Va"]):
        plot_accs(values, var, l)
    decore_plot(ax)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.1, top=0.93, wspace=0.2, hspace=0.35)
    plt.savefig("../plots/best_active_learning.pdf")
    plt.show()
