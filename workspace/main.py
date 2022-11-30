import tensorflow as tf

xtrain = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
ytrain = [1, 1, 0, 1, 2, 0]

train_imgs, train_lbls = [], []
for i in range(5):
    rand = tf.random.uniform(shape=[len(xtrain)], minval=0, maxval=len(xtrain) - 1, dtype=tf.dtypes.int64)
    train_imgs.append(tf.convert_to_tensor([xtrain[index] for index in rand]))
    train_lbls.append(tf.convert_to_tensor([ytrain[index] for index in rand]))

print(train_imgs, train_lbls)

train_imgs, train_lbls = [], []
for i in range(5):
    rand = tf.random.uniform(shape=[len(xtrain)], minval=0, maxval=len(xtrain) - 1, dtype=tf.dtypes.int64)
    train_imgs.append(tf.gather(xtrain, rand))
    train_lbls.append(tf.gather(ytrain, rand))

print(train_imgs, train_lbls)