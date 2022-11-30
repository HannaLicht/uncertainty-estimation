from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tqdm
from matplotlib import pyplot as plt
import tensorflow as tf
from uncertainty.metrics_classification import reliability_diagram


class NeighborhoodUncertaintyClassifier:

    k = 10

    def __init__(self, model, xtrain, ytrain, xtest, ytest, x=None, path_uncertainty_model=None):
        """
        :param x: images for which the uncertainty should be estimated
        :param y: labels of these images if known (for evaluation)
        """
        self.model = model
        self.ytrain = ytrain
        self.ytest = ytest
        self.x = xtest if x is None else x
        self.y = ytest if x is None else None
        self.dataset_train = tf.data.Dataset.from_tensor_slices((xtrain, self.ytrain)).batch(100)
        self.dataset_test = tf.data.Dataset.from_tensor_slices((xtest, self.ytest)).batch(100)
        output = self.model.layers[-2].output
        self.model_without_last_layer = tf.keras.Model(inputs=self.model.input, outputs=output)
        self.model_without_last_layer.compile()
        self.A = self.model_without_last_layer.predict(xtrain)
        try:
            self.build_uncertainty_model()
            self.uncertainty_model.load_weights(path_uncertainty_model)
        except:
            self.__train_uncertainty_model(path_uncertainty_model)
        self.certainties = self.get_certainties()

    def build_uncertainty_model(self):
        inp = tf.keras.Input((2*self.k + 1))
        x = tf.keras.layers.Dense(512, activation='relu')(inp)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        self.uncertainty_model = tf.keras.Model(inputs=inp, outputs=out)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004)
        self.uncertainty_model.compile(optimizer=optimizer,
                                       loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                                       metrics=[tf.keras.metrics.BinaryAccuracy()])

    def __get_inputs_uncertainty_model(self, out, img_batch, train_data: bool):
        ypred = tf.argmax(out, axis=-1)
        ypred = tf.reshape(tf.repeat(ypred, 10, axis=0), (-1, 10))
        spred = tf.math.reduce_max(out, axis=-1)
        r = self.model_without_last_layer.predict(img_batch, verbose=0)
        r = tf.repeat(r, int(len(self.A) / 5), axis=0)
        r = tf.reshape(r, (len(img_batch), int(len(self.A) / 5), -1))

        distances = tf.reduce_sum(tf.abs(tf.subtract(self.A[:int(len(self.A) / 5)], r)), axis=-1)
        distances = distances + tf.reduce_sum(tf.abs(tf.subtract(
            self.A[int(len(self.A) / 5):int(2 * len(self.A) / 5)], r)), axis=-1)
        distances = distances + tf.reduce_sum(tf.abs(tf.subtract(
            self.A[int(2 * len(self.A) / 5):int(3 * len(self.A) / 5)], r)), axis=-1)
        distances = distances + tf.reduce_sum(tf.abs(tf.subtract(
            self.A[int(3 * len(self.A) / 5):int(4 * len(self.A) / 5)], r)), axis=-1)
        distances = distances + tf.reduce_sum(tf.abs(tf.subtract(
            self.A[int(4 * len(self.A) / 5):len(self.A)], r)), axis=-1)
        # distances = tf.reduce_sum(tf.abs(tf.subtract(self.A[:len(self.A/5)], r)), axis=-1)
        if train_data:
            # the smallest distance has the r itself -> get k+1 vectors and remove the best one
            top_k_distances, top_k_indices = tf.nn.top_k(tf.negative(distances), k=self.k + 1)
            top_k_distances = tf.slice(top_k_distances, [0, 1], [len(img_batch), self.k])
            top_k_indices = tf.slice(top_k_indices, [0, 1], [len(img_batch), self.k])
        else:
            top_k_distances, top_k_indices = tf.nn.top_k(tf.negative(distances), k=self.k)
        top_k_labels = tf.gather(self.ytrain, top_k_indices)
        top_k_agreed = tf.cast(tf.math.equal(ypred, tf.cast(top_k_labels, tf.int64)), tf.float32)

        inputs = tf.concat([top_k_distances, top_k_agreed, tf.reshape(spred, (-1, 1))], axis=-1)
        return inputs

    def __get_data_for_uncertainty_model(self, train: bool):
        x_uncertainty = []
        y_uncertainty = []
        dataset = self.dataset_train if train else self.dataset_test

        for _, (img_batch, lbl_batch) in zip(tqdm.tqdm(range(len(dataset))), dataset):
            out = self.model.predict(img_batch, verbose=0)
            inputs = self.__get_inputs_uncertainty_model(out, img_batch, train)
            x_uncertainty.append(inputs)
            y_uncertainty.append(tf.equal(lbl_batch, tf.argmax(out, axis=-1)))

        x_uncertainty = tf.reshape(x_uncertainty, (-1, 2*self.k + 1))
        y_uncertainty = tf.cast(tf.reshape(y_uncertainty, (-1)), tf.float32)
        return x_uncertainty, y_uncertainty

    def __train_uncertainty_model(self, path_uncertainty_model):
        self.build_uncertainty_model()

        # prepare datasets for uncertainty model
        xtrain_uncertainty, ytrain_uncertainty = self.__get_data_for_uncertainty_model(train=True)
        xtest_uncertainty, ytest_uncertainty = self.__get_data_for_uncertainty_model(train=False)

        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)
        rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)
        self.uncertainty_model.fit(xtrain_uncertainty, ytrain_uncertainty,
                                   validation_data=(xtest_uncertainty, ytest_uncertainty),
                                   callbacks=[early_stop, rlrop],
                                   epochs=1000)
        if path_uncertainty_model is not None:
            self.uncertainty_model.save_weights(path_uncertainty_model)

        # test if weights of best val_loss are saved --> yes :)
        self.build_uncertainty_model()
        self.uncertainty_model.load_weights(path_uncertainty_model)

        loss, acc = self.uncertainty_model.evaluate(xtest_uncertainty, ytest_uncertainty, verbose=2)
        print("Uncertainty model, accuracy: {:5.2f}%".format(100 * acc))

    def predict_certainty(self, test_img_batch):
        out = self.model.predict(test_img_batch, verbose=0)
        inputs = self.__get_inputs_uncertainty_model(out, test_img_batch, False)
        u = self.uncertainty_model.predict(inputs, verbose=0)
        return u

    def get_certainties(self):
        x_batched = tf.data.Dataset.from_tensor_slices(self.x).batch(100)
        certainties = []
        for _, img_batch in zip(tqdm.tqdm(range(len(x_batched))), x_batched):
            certainties.append(self.predict_certainty(img_batch))
        return tf.reshape(certainties, (len(self.x))).numpy()

    def certainty_score(self):
        pred_y = tf.math.argmax(self.model.predict(self.x), axis=-1)
        correct = (pred_y == self.y)
        pred_true, pred_false = [], []
        for i in range(len(self.certainties)):
            pred_true.append(self.certainties[i]) if correct[i] else pred_false.append(self.certainties[i])
        score = len(pred_false)*tf.math.reduce_sum(pred_true)/(len(pred_true)*tf.math.reduce_sum(pred_false))
        return score.numpy()

    def plot_diagrams(self):
        out = self.model.predict(self.x)
        pred_y = tf.math.argmax(out, axis=-1)
        correct = (pred_y == self.y)

        plt.figure(figsize=(10, 5))
        # reliability diagrams
        plt.subplot(1, 2, 1)
        reliability_diagram(self.y, out, self.certainties)

        pred_true, pred_false = [], []
        for i in range(len(correct)):
            if correct[i]:
                pred_true.append(self.certainties[i])
            else:
                pred_false.append(self.certainties[i])

        plt.subplot(1, 2, 2)
        plt.boxplot([pred_true, pred_false], showfliers=False)
        plt.ylim(-0.05, 1.05)
        plt.xticks([1.0, 2.0], ["correct", "incorrect"])
        plt.xlabel("Predictions")
        plt.ylabel("Certainty")

        plt.suptitle("Neighborhood Uncertainty", fontsize=14)
        plt.show()