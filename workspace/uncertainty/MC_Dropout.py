# https://www.depends-on-the-definition.com/model-uncertainty-in-deep-learning-with-monte-carlo-dropout/
# https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
from uncertainty.metrics_classification import reliability_diagram, mutual_information
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
            m.update_state(tf.keras.utils.to_categorical(lbls, self.num_classes), y_p)
            accs.append(m.result())
        return (sum(accs)/len(accs)).numpy()

    def get_ensemble_accuracy(self, lbls):
        mc_ensemble_pred = tf.argmax(self.p_ens, axis=-1)
        en_acc = tf.keras.metrics.Accuracy()
        en_acc.update_state(lbls, mc_ensemble_pred)
        return en_acc.result().numpy()

    # select uncertain images by the standard deviation of the class predictions
    def get_certainties_by_stddev(self):
        stddevs = tf.math.sqrt(tf.math.reduce_mean(tf.math.reduce_variance(self.predictions, axis=0), axis=-1)).numpy()
        max_var = ((1/self.num_classes)**2*(self.num_classes-1) +
                   ((self.num_classes-1)/self.num_classes)**2)/(self.num_classes+1)
        max_stddevs = tf.math.sqrt(max_var).numpy()
        uncertainty = stddevs/max_stddevs
        return 1-uncertainty

    def get_certainties_by_SE(self):
        H = tfd.Categorical(probs=self.p_ens).entropy()
        H = H / -tf.math.log(1/self.num_classes)
        return 1-H.numpy()

    def get_certainties_by_mutual_inf(self):
        mi = mutual_information(self.predictions)
        mi = mi / -tf.math.log(1/self.num_classes)
        return 1-mi.numpy()

    def get_ensemble_prediction(self):
        mc_ensemble_pred = tf.argmax(self.p_ens, axis=1)
        return mc_ensemble_pred.numpy()

    def certainty_scores(self, lbls):
        pred_y = tf.math.argmax(self.p_ens, axis=-1)
        correct = (pred_y == lbls)

        scores = []
        for certainties in (self.get_certainties_by_SE(), self.get_certainties_by_mutual_inf()):
            pred_true, pred_false = [], []
            for i in range(len(certainties)):
                pred_true.append(certainties[i]) if correct[i] else pred_false.append(certainties[i])
            score = len(pred_false)*tf.math.reduce_sum(pred_true)/(len(pred_true)*tf.math.reduce_sum(pred_false))
            scores.append(score.numpy())

        return scores

    def plot_diagrams(self, lbls):
        pred_y = tf.math.argmax(self.p_ens, axis=-1)
        correct = (pred_y == lbls)
        plt.figure(figsize=(12, 8))
        cert_se = self.get_certainties_by_SE()
        cert_stddev = self.get_certainties_by_mutual_inf()

        # reliability diagrams
        plt.subplot(2, 2, 1)
        reliability_diagram(lbls, self.p_ens, cert_se, method="Shannon Entropy")
        plt.subplot(2, 2, 2)
        reliability_diagram(lbls, self.p_ens, cert_stddev, method="Mutual Information")

        pred_true_u1, pred_false_u1, pred_true_u2, pred_false_u2 = [], [], [], []
        for i in range(len(correct)):
            if correct[i]:
                pred_true_u1.append(cert_se[i])
                pred_true_u2.append(cert_stddev[i])
            else:
                pred_false_u1.append(cert_se[i])
                pred_false_u2.append(cert_stddev[i])

        plt.subplot(2, 2, 3)
        plt.boxplot([pred_true_u1, pred_false_u1, pred_true_u2, pred_false_u2], showfliers=False)
        plt.ylim(-0.05, 1.05)
        plt.xticks([1.0, 2.0, 3.0, 4.0], ["correct SE", "incorrect SE", "correct MI", "incorrect MI"])
        plt.xlabel("Predictions")
        plt.ylabel("Certainty")

        plt.subplot(2, 2, 4)
        plt.scatter(pred_true_u1, pred_true_u2, c="limegreen", s=10, label="true predictions")
        plt.scatter(pred_false_u1, pred_false_u2, c="firebrick", s=10, label="wrong predictions")
        plt.xlabel("SE Certainty")
        plt.ylabel("MI Certainty")
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.05, 1.05)
        plt.legend(loc="lower left")

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

