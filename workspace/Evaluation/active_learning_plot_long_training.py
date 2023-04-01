import json
import matplotlib.pyplot as plt
import numpy as np
from functions import COLORS


with open('../Results/active_learning/long_training.json') as json_file:
    results = json.load(json_file)

x = np.linspace(200, 25000, 249)

for key, c in zip(results, [COLORS["NUC Va"], COLORS["MCD MI"], "gray"]):
    plt.plot(x, results[key], label=key if key != "random" else "Zufall", linewidth=1.1, color=c)

plt.xlabel("Anzahl gelabelter Bilder")
plt.ylabel("Testaccuracy")
plt.grid(visible=True, color="gainsboro", linestyle='dashed', zorder=0)
plt.legend(loc="lower right")
plt.ylim(0.471, 0.74)

plt.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98, wspace=0.25, hspace=0.35)
plt.savefig("../Plots/long_training.pdf")
plt.show()
