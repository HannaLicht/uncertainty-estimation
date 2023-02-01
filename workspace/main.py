import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
methods = ["Softmax"]
certainties = tf.convert_to_tensor([[0.1, 0.4, 0.6, 0.3, 0.8, 0.7, 0.2, 0.15, 0.05, 0.46, 0.72]])

for certs, method in zip(certainties, methods):
    TU, FC = [], []
    for thr in thresholds:
        count = 0
        for cert in certs:
            if cert < thr:
                count = count+1
        TU.append(count)
        FC.append(len(certs)-count)
    recall = tf.divide(TU, tf.add(TU, FC)).numpy()
    with open('Evaluation/recalls_ood_data_.json') as json_file:
        data = json.load(json_file)
        for i in range(len(thresholds)):
            data[method][i] = data[method][i] + [recall[i].item()]
    with open('Evaluation/recalls_ood_data_.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    plt.plot(thresholds, recall, label=method)

plt.legend(loc="lower right")
plt.show()
