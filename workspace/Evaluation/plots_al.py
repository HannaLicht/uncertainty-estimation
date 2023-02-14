import json
import matplotlib.pyplot as plt
import tensorflow as tf
from functions import adjust_lightness, COLORS

STARTDATA = 1000
NUM_IMAGES = 100
HIGHER_DIVERSITY = False


if STARTDATA == 100:
    times_images_added = 19
elif STARTDATA == 10000 or (STARTDATA == 1000 and NUM_IMAGES == 100):
    times_images_added = 10
else:
    times_images_added = 9
#results = "just uncertainty" if not HIGHER_DIVERSITY else "diversity"


with open('../Results/active_learning/' + str(STARTDATA) + "_" + str(NUM_IMAGES) + ".json") as json_file:
    data = json.load(json_file)
    #results = data["diversity"] if HIGHER_DIVERSITY else data["just uncertainty"]
    results = data

numbers = [STARTDATA + (i+1)*NUM_IMAGES for i in range(times_images_added)]

rand = [tf.reduce_mean(results[str(imgs)]["random"]) for imgs in numbers]
softmax = [tf.reduce_mean(results[str(imgs)]["softmax_entropy"]) for imgs in numbers]
mc_se = [tf.reduce_mean(results[str(imgs)]["MC_drop"]["SE"]) for imgs in numbers]
mc_mi = [tf.reduce_mean(results[str(imgs)]["MC_drop"]["MI"]) for imgs in numbers]
bag_se = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["BaggingEns"]["SE"]) for imgs in numbers]
bag_mi = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["BaggingEns"]["MI"]) for imgs in numbers]
aug_se = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["SE"]) for imgs in numbers]
aug_mi = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["MI"]) for imgs in numbers]
ris_se = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["RandomInitShuffleEns"]["SE"]) for imgs in numbers]
ris_mi = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["RandomInitShuffleEns"]["MI"]) for imgs in numbers]
nuc = [tf.reduce_mean(results[str(imgs)]["NUC"]) for imgs in numbers]
nuc_tr = [tf.reduce_mean(results[str(imgs)]["NUC Tr"]) for imgs in numbers]

rand_var = [tf.math.reduce_std(results[str(imgs)]["random"]) for imgs in numbers]
softmax_var = [tf.math.reduce_std(results[str(imgs)]["softmax_entropy"]) for imgs in numbers]
mc_se_var = [tf.math.reduce_std(results[str(imgs)]["MC_drop"]["SE"]) for imgs in numbers]
mc_mi_var = [tf.math.reduce_std(results[str(imgs)]["MC_drop"]["MI"]) for imgs in numbers]
bag_se_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["BaggingEns"]["SE"]) for imgs in numbers]
bag_mi_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["BaggingEns"]["MI"]) for imgs in numbers]
aug_se_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["SE"]) for imgs in numbers]
aug_mi_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["MI"]) for imgs in numbers]
ris_se_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["RandomInitShuffleEns"]["SE"]) for imgs in numbers]
ris_mi_var = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["RandomInitShuffleEns"]["MI"]) for imgs in numbers]
nuc_var = [tf.math.reduce_std(results[str(imgs)]["NUC"]) for imgs in numbers]
nuc_tr_var = [tf.math.reduce_std(results[str(imgs)]["NUC Tr"]) for imgs in numbers]


plt.figure(figsize=(11.5, 3.5))

'''if STARTDATA == 100 and NUM_IMAGES == 100:
    methods_to_show = [rand, aug_mi, bag_mi, bag_se, aug_se]
    vars_methods = [rand_var, aug_mi_var, bag_mi_var,
                    bag_se_var, aug_se_var]
    labels = ["Zufall", "Data Aug. MI", "Bagging MI",
              "Bagging SE", "Data Aug. SE"]

elif STARTDATA == 1000 and NUM_IMAGES == 1000:
    methods_to_show = [rand, nuc_tr, nuc, mc_mi, ris_mi, softmax, bag_mi, aug_mi, mc_se, bag_se, ris_se, aug_se]
    labels = ["Zufall", "NUC Tr" "NUC Va", "MC Dropout MI", "ZIS MI", "Softmax SE", "Bagging MI", "Data Aug. MI",
              "MC Dropout SE", "Bagging SE", "ZIS SE", "Data Aug. SE"]
    plt.ylim(54.5, 67.8)

elif STARTDATA == 1000 and NUM_IMAGES == 100:
    methods_to_show = [rand, nuc_tr, nuc, softmax, mc_mi, ris_mi, bag_mi, aug_mi, mc_se, ris_se, aug_se, bag_se]
    labels = ["Zufall", "NUC Tr", "NUC Va", "Softmax SE", "MC Dropout MI", "ZIS MI", "Bagging MI", "Data Aug. MI",
              "MC Dropout SE", "ZIS SE", "Data Aug. SE", "Bagging SE"]
    elif STARTDATA == 100 and NUM_IMAGES == 10:
    methods_to_show = [rand, nuc, aug_mi, mc_mi, ris_mi, softmax, ris_se, bag_mi, bag_se, mc_se, aug_se]
    labels = ["Zufall", "NUC", "Data Aug. MI", "MC Dropout MI", "ZIS MI", "Softmax SE", "ZIS SE", "Bagging MI",
              "Bagging SE", "MC Dropout SE", "Data Aug. SE"]

    elif STARTDATA == 1000 and NUM_IMAGES == 10:
    methods_to_show = [rand, mc_mi, nuc, softmax, mc_se, bag_mi, aug_mi, ris_mi, bag_se, ris_se, aug_se]
    labels = ["Zufall", "MC Dropout MI", "NUC", "Softmax SE", "MC Dropout SE", "Bagging MI", "Data Aug. MI", "ZIS MI",
              "Bagging SE", "ZIS SE", "Data Aug. SE"]
else:
    methods_to_show = [rand, mc_mi, nuc, softmax, mc_se, bag_mi, aug_mi, ris_mi, bag_se, ris_se, aug_se]
    labels = ["Zufall", "MC Dropout MI", "NUC", "Softmax SE", "MC Dropout SE", "Bagging MI", "Data Aug. MI", "ZIS MI",
              "Bagging SE", "ZIS SE", "Data Aug. SE"]
    plt.ylim(67.6, 71.6)

colors = ["black", "blue", adjust_lightness("green", 1.3), adjust_lightness("green", 1.6), "yellowgreen",
              adjust_lightness("olive", 1.5), adjust_lightness("yellow", 0.9), "gold", "orange",
              adjust_lightness("coral", 0.9), adjust_lightness("red", 1.1), adjust_lightness("red", 0.8)
              ]
'''


def plot_accs(values, var, l):
    c = COLORS[l] if l != "Zufall" else "black"
    plt.plot(numbers, values, "--" if l == "Zufall" else "-", label=l, color=c, zorder=1, linewidth=1.2)
    plt.fill_between(numbers, tf.math.add(values, var), tf.math.subtract(values, var), color=c, alpha=0.1, zorder=0)
    if l == "Zufall":
        plt.plot(numbers, values, label=" ", color="white", zorder=0)


def decore_plot(ax):
    plt.xlabel("Anzahl gelabelter Bilder")
    plt.ylabel("Testaccuracy")
    plt.grid(visible=True, color="gainsboro", linestyle='dashed', zorder=0)
    ax.set_axisbelow(True)
    plt.legend(loc="lower right")
    if STARTDATA == 100:
        plt.ylim(0.26, 0.57)
    elif STARTDATA == 1000 and NUM_IMAGES == 100:
        plt.ylim(0.508, 0.565)


ax = plt.subplot(1, 3, 1)
for values, var, l in zip([rand, bag_se, bag_mi, aug_se, aug_mi],
                       [rand_var, bag_se_var, bag_mi_var, aug_se_var, aug_mi_var],
                       ["Zufall", "Bag SE", "Bag MI", "DA SE", "DA MI"]):
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
div = "_raised_diversity" if HIGHER_DIVERSITY else ""
plt.savefig("../plots/active_learning_" + str(STARTDATA) + "_" + str(NUM_IMAGES) + div + ".pdf")

plt.show()