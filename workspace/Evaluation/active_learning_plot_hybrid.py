import json
import matplotlib.pyplot as plt
import tensorflow as tf
from functions import COLORS


TIMES_IMAGES_ADDED = 10


file_names = ["100_100", "1000_100", "1000_1000", "10000_1000"]
titles = ["Start: 100  Hinzu: 100", "Start: 1000  Hinzu: 100", "Start: 1000  Hinzu: 1000",
          "Start: 10000  Hinzu: 1000"]

startdata = [100, 1000, 1000, 10000]
num_images = [100, 100, 1000, 1000]
plt.figure(figsize=(9, 8))


for count, (file, title, start, num) in enumerate(zip(file_names, titles, startdata, num_images)):

    results = json.load(open('../Results/active_learning/' + file + "_hybrid.json"))
    ax = plt.subplot(2, 2, count+1)
    plt.title(title)
    plt.xlabel("Anzahl gelabelter Bilder")
    plt.ylabel("Testaccuracy")
    plt.grid(visible=True, color="gainsboro", linestyle='dashed', zorder=0)
    ax.set_axisbelow(True)

    numbers = [start + (i+1)*num for i in range(TIMES_IMAGES_ADDED)]
    rand = [tf.reduce_mean(json.load(open('../Results/active_learning/' + file + '.json'))[str(imgs)]["random"])
            for imgs in numbers]
    # div = [tf.reduce_mean(results[str(imgs)]["just_divers"]) for imgs in numbers]
    nuc_va = [tf.reduce_mean(results[str(imgs)]["NUC Va"]) for imgs in numbers]
    mcd_mi = [tf.reduce_mean(results[str(imgs)]["MC_drop"]["MI"]) for imgs in numbers]
    da_mi = [tf.reduce_mean(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["MI"]) for imgs in numbers]

    r_v = [tf.math.reduce_std(json.load(open('../Results/active_learning/' + file + '.json'))[str(imgs)]["random"])
           for imgs in numbers]
    # div_v = [tf.math.reduce_std(results[str(imgs)]["just_divers"]) for imgs in numbers]
    nuc_va_v = [tf.math.reduce_std(results[str(imgs)]["NUC Va"]) for imgs in numbers]
    mcd_mi_v = [tf.math.reduce_std(results[str(imgs)]["MC_drop"]["MI"]) for imgs in numbers]
    da_mi_v = [tf.math.reduce_std(results[str(imgs)]["Ensembles"]["DataAugmentationEns"]["MI"]) for imgs in numbers]

    plt.plot(numbers, rand, label="Zufall", color="black", linestyle="--", linewidth=1.2, zorder=1)
    plt.plot(numbers, rand, label=" ", color="white", zorder=0)
    # plt.plot(numbers, div, label="max. Diversität", color="purple", linestyle="--", linewidth=1.2, zorder=1)
    plt.plot(numbers, nuc_va, label="NUC Va", color=COLORS["NUC Va"], linewidth=1.2, zorder=1)
    plt.plot(numbers, mcd_mi, label="MCD MI", color=COLORS["MCD MI"], linewidth=1.2, zorder=1)
    plt.plot(numbers, da_mi, label="DA MI", color=COLORS["DA MI"], linewidth=1.2, zorder=1)

    plt.fill_between(numbers, tf.math.add(rand, r_v), tf.math.subtract(rand, r_v), color="black", alpha=0.05,
                     zorder=0)
    # plt.fill_between(numbers, tf.math.add(div, div_v), tf.math.subtract(div, div_v), color="purple", alpha=0.07,
    #                zorder=0)
    plt.fill_between(numbers, tf.math.add(nuc_va, nuc_va_v), tf.math.subtract(nuc_va, nuc_va_v),
                     color=COLORS["NUC Va"], alpha=0.1, zorder=0)
    plt.fill_between(numbers, tf.math.add(mcd_mi, mcd_mi_v), tf.math.subtract(mcd_mi, mcd_mi_v),
                     color=COLORS["MCD MI"], alpha=0.065, zorder=0)
    plt.fill_between(numbers, tf.math.add(da_mi, da_mi_v), tf.math.subtract(mcd_mi, mcd_mi_v),
                     color=COLORS["DA MI"], alpha=0.1, zorder=0)
    plt.legend(loc="lower right")

plt.subplots_adjust(left=0.07, right=0.98, bottom=0.06, top=0.95, wspace=0.25, hspace=0.3)
plt.savefig("../plots/active_learning_hybrid.pdf")
plt.show()