# https://www.depends-on-the-definition.com/model-uncertainty-in-deep-learning-with-monte-carlo-dropout/
# https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
from uncertainty.metrics_classification import reliability_diagram, uncertainty_diagram
import re
import tensorflow_probability as tfp
tfd = tfp.distributions


class SamplingBasedEstimator:

    predictions = []
    eval_predictions = []
    X = []
    p_ens = []
    num_classes = None
    estimator_name = ""

    def get_simple_model_accuracy(self, lbls):
        accs = []
        m = tf.keras.metrics.CategoricalAccuracy()
        for y_p in self.predictions:
            m.reset_state()
            m.update_state(lbls, y_p)
            accs.append(m.result())
        return (sum(accs)/len(accs)).numpy()

    def get_ensemble_accuracy(self, lbls):
        mc_ensemble_pred = tf.argmax(self.p_ens, axis=-1)
        en_acc = tf.keras.metrics.Accuracy()
        en_acc.update_state(tf.argmax(lbls, axis=-1), mc_ensemble_pred)
        return en_acc.result().numpy()

    def get_ensemble_prediction(self):
        return tf.argmax(self.p_ens, axis=-1).numpy()

    def uncertainties_shannon_entropy(self):
        return tfd.Categorical(probs=self.p_ens).entropy().numpy()

    def uncertainties_mutual_information(self):
        h = tfd.Categorical(probs=self.p_ens).entropy()
        mi = 0
        for prediction in self.predictions:
            mi = mi - tfd.Categorical(probs=prediction).entropy()
        mi = mi / len(self.predictions) + h
        return mi

    def bounded_certainties_shannon_entropy(self):
        h = tfd.Categorical(probs=self.p_ens).entropy()
        h = h / -tf.math.log(1/self.num_classes)
        return 1-h.numpy()

    def bounded_certainties_mutual_information(self):
        mi = self.uncertainties_mutual_information()
        mi = mi / -tf.math.log(1/self.num_classes)
        return 1-mi.numpy()

    def certainty_scores(self, lbls):
        pred_y = tf.math.argmax(self.p_ens, axis=-1)
        correct = (pred_y == lbls)

        scores = []
        for uncertainties in (self.uncertainties_shannon_entropy(), self.uncertainties_mutual_information()):
            pred_true, pred_false = [], []
            for i in range(len(uncertainties)):
                pred_true.append(uncertainties[i]) if correct[i] else pred_false.append(uncertainties[i])
            score = len(pred_true)*tf.math.reduce_sum(pred_false)/(len(pred_false)*tf.math.reduce_sum(pred_true))
            scores.append(score.numpy())
        return scores

    def plot_diagrams(self, lbls):
        pred_y = tf.math.argmax(self.p_ens, axis=-1)
        correct = (pred_y == lbls)
        plt.figure(figsize=(16, 10))
        cert_se = self.bounded_certainties_shannon_entropy()
        cert_mi = self.bounded_certainties_mutual_information()

        # calibration diagrams
        plt.subplot(2, 3, 1)
        reliability_diagram(lbls, self.p_ens, cert_se, method="Shannon Entropy")
        reliability_diagram(lbls, self.p_ens, cert_mi, method="Mutual Information")

        pred_true_se, pred_false_se, pred_true_mi, pred_false_mi = [], [], [], []
        uncert_se = self.uncertainties_shannon_entropy()
        uncert_mi = self.uncertainties_mutual_information()
        for i in range(len(correct)):
            if correct[i]:
                pred_true_se.append(uncert_se[i])
                pred_true_mi.append(uncert_mi[i])
            else:
                pred_false_se.append(uncert_se[i])
                pred_false_mi.append(uncert_mi[i])

        plt.subplot(2, 3, 2)
        plt.boxplot([pred_true_se, pred_false_se], showfliers=False)
        plt.xticks([1.0, 2.0], ["correct SE", "incorrect SE"])
        plt.xlabel("Predictions")
        plt.ylabel("Uncertainty")

        plt.subplot(2, 3, 3)
        plt.boxplot([pred_true_mi, pred_false_mi], showfliers=False)
        plt.xticks([1.0, 2.0], ["correct MI", "incorrect MI"])
        plt.xlabel("Predictions")
        plt.ylabel("Uncertainty")

        plt.subplot(2, 3, 4)
        plt.scatter(pred_true_se, pred_true_mi, c="limegreen", s=10, label="true predictions")
        plt.scatter(pred_false_se, pred_false_mi, c="firebrick", s=10, label="wrong predictions")
        plt.xlabel("SE Uncertainty")
        plt.ylabel("MI Uncertainty")
        plt.legend(loc="upper left")

        plt.subplot(2, 3, 5)
        uncertainty_diagram(lbls, self.p_ens, uncert_se, method="Shannon Entropy")
        plt.subplot(2, 3, 6)
        uncertainty_diagram(lbls, self.p_ens, uncert_mi, method="Mutual Information")

        plt.suptitle(self.estimator_name, fontsize=14)
        plt.show()


class MCDropoutEstimator(SamplingBasedEstimator):

    def __init__(self, model, X, num_classes=10, T=50):
        """
        :param X: data for which the uncertainty should be estimated
        :param T: number of runs with activated dropout at inference time
        """
        self.T = T
        self.estimator_name = "MC Dropout"
        self.model = self.make_MC_dropout(model, layer_regex=".*drop.*")

        # check, if all dropout layers are MC dropout layers (training=True -> activated during inference)
        for layer in self.model.get_config().get("layers"):
            print(layer)

        self.X, self.num_classes, self.predictions = X, num_classes, []
        X = [X[i:i + 1000] for i in range(0, len(X), 1000)]
        for _ in tqdm.tqdm(range(self.T)):
            preds = self.model(X[0])[-1]
            for img_batch in X[1:]:
                preds = tf.concat([preds, self.model(img_batch)[-1]], axis=0)
            self.predictions.append(preds)
        self.p_ens = tf.math.reduce_mean(self.predictions, axis=0)

    def make_MC_dropout(self, model, layer_regex):
        # Auxiliary dictionary to describe the network graph
        network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

        # Set the input layers of each layer
        for layer in model.layers:
            for node in layer._outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in network_dict['input_layers_of']:
                    network_dict['input_layers_of'].update({layer_name: [layer.name]})
                else:
                    network_dict['input_layers_of'][layer_name].append(layer.name)

        # Set the output tensor of the input layer
        network_dict['new_output_tensor_of'].update({model.layers[0].name: model.input})

        # Iterate over all layers after the input
        model_outputs = []
        for layer in model.layers[1:]:

            # Determine input tensors
            layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                           for layer_aux in network_dict['input_layers_of'][layer.name]]
            if len(layer_input) == 1:
                layer_input = layer_input[0]

            # Insert layer if name matches the regular expression
            if re.match(layer_regex, layer.name):
                # training=True important to enable active dropout when predicting -> MC Monte Carlo
                x = layer(layer_input, training=True)
                # print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name, layer.name, position))
            else:
                x = layer(layer_input)

            # Set new output tensor (the original one, or the one of the inserted layer)
            network_dict['new_output_tensor_of'].update({layer.name: x})

            # Save tensor in output list if it is output in initial model
            if layer_name in model.output_names:
                model_outputs.append(x)

        return tf.keras.Model(inputs=model.inputs, outputs=model_outputs)

