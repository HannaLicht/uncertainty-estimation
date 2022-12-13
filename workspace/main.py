import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

predictions = [[[[0.5, 0.4, 0.1], [0.33, 0.33, 0.34], [0.0001, 0.0001, 0.9998]],
         [[0.001, 0.001, 0.998], [0.3, 0.2, 0.5], [0.4, 0.3, 0.3]]],
               [[[0.5, 0.4, 0.1], [0.33, 0.33, 0.34], [0.0001, 0.0001, 0.9998]],
         [[0.001, 0.998, 0.001], [0.8, 0.1, 0.1], [0.45, 0.45, 0.1]]],
               [[[0.5, 0.4, 0.1], [0.33, 0.33, 0.34], [0.0001, 0.0001, 0.9998]],
         [[0.998, 0.001, 0.001], [0.6, 0.2, 0.2], [0.1, 0.05, 0.85]]]]

p_ens = tf.reduce_mean(predictions, axis=0)
print(p_ens)

h = tfd.Categorical(probs=p_ens).entropy()
mi = 0
for prediction in predictions:
    mi = mi - tfd.Categorical(probs=prediction).entropy()
mi = mi / len(predictions) + h

print(mi)

mean_mi = tf.reduce_mean(mi, axis=-1)
print(mean_mi)