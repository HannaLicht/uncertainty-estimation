import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

plt.figure(figsize=(4.5, 3.7))
plt.plot([0, 1], [0, 1], 'k--', label="Perfekt kalibriert")
plt.plot(np.linspace(start=0, stop=0.5, num=50), tf.sigmoid(np.linspace(start=-6, stop=0, num=50)),
         label="Zu selbstsicher", color="red")
plt.plot(np.linspace(start=0.5, stop=1, num=50), tf.sigmoid(np.linspace(start=0, stop=6, num=50)),
         label="Zu unsicher", color="brown")
plt.fill_between(np.linspace(start=0, stop=0.5, num=50), np.linspace(start=0, stop=0.5, num=50),
                 tf.sigmoid(np.linspace(start=-6, stop=0, num=50)), color='red', alpha=0.15)
plt.fill_between(np.linspace(start=0.5, stop=1, num=50), np.linspace(start=0.5, stop=1, num=50),
                 tf.sigmoid(np.linspace(start=0, stop=6, num=50)), color='brown', alpha=0.15)
plt.xlabel('Certainty')
plt.ylabel('Accuracy')
plt.legend(loc="upper left")
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95, wspace=0.3, hspace=0.35)

plt.savefig("../plots/calibration_plot.png", dpi=300)
plt.show()
