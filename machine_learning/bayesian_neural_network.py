from dataclasses import dataclass
from typing import List, Union
from pathlib import Path
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pylab as plt

from .neural_networks import NeuralNetwork


@dataclass
class BayesianNeuralNetwork(NeuralNetwork):
    """
    Class of the Bayesian Neural Network model. This default model is based on the
    tutorial of the tensorflow probability package found in
    https://keras.io/examples/keras_recipes/bayesian_neural_networks/


    :param nb_neurons: Number of neurons in each hidden layer
    :param learning_rate: The learning rate of the default optimization technique

    """

    nb_neurons: Union[List, None, np.array] = None
    learning_rate: float = 0.0001

    @staticmethod
    def prior(kernel_size, bias_size, dtype=None):
        """
        Define the prior weight distribution as Normal of mean=0 and stddev=1.
        Note that, in this example, the we prior distribution is not trainable,
        as we fix its parameters.

        """
        n = kernel_size + bias_size
        prior_model = keras.Sequential(
            [
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.MultivariateNormalDiag(
                        loc=tf.zeros(n), scale_diag=tf.ones(n)
                    )
                )
            ]
        )
        return prior_model

    @staticmethod
    def posterior(kernel_size, bias_size, dtype=None):
        """
        Define variational posterior weight distribution as multivariate Gaussian.
        Note that the learnable parameters for this distribution are the means,
        variances, and covariances.
        """
        n = kernel_size + bias_size
        posterior_model = keras.Sequential(
            [
                tfp.layers.VariableLayer(
                    tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
                ),
                tfp.layers.MultivariateNormalTriL(n),
            ]
        )
        return posterior_model

    def __build_model(self) -> None:
        """
        Function that builds the BNN

        """
        self.probabilistic = True
        inputs = keras.Input(shape=(self.training_data.shape[1],))
        features = layers.BatchNormalization()(inputs)
        # hidden layers
        for i in range(self.nb_hidden_layers):
            features = tfp.layers.DenseVariational(
                units=self.nb_neurons[i],
                make_prior_fn=BayesianNeuralNetwork.prior,
                make_posterior_fn=BayesianNeuralNetwork.posterior,
                kl_weight=1 / self.training_data.shape[0],
                activation=self.activation_fct.value,
            )(features)

        units_output = self.target.shape[1] * 2
        # Create a probabilistic output (Normal distribution), and use the `Dense` layer
        # to produce the parameters of the distribution.
        # We set units=2 * targets to learn both the mean and the variance of the Normal distribution.
        distribution_params = layers.Dense(units=units_output)(features)
        outputs = tfp.layers.IndependentNormal(self.target.shape[1])(
            distribution_params
        )

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.summary()
        return

    @staticmethod
    def negative_loglikelihood(targets, estimated_distribution):
        """
        Since the output of the model is a distribution, rather than a point estimate, we use the negative loglikelihood as our loss function to compute how likely to see the true data (targets) from the estimated distribution produced by the model.
        """
        return -estimated_distribution.log_prob(targets)

    def train_classification(self) -> None:
        """
        Method that trains a BNN classification model.
        """
        raise ValueError(
            " A Bayesian neural network supports only the regression process."
        )
        return

    def train_regression(self) -> None:
        """
        Method that trains a BNN regression model.
        """
        # scale training data
        scaled_training_data = self.rescale_training_data()
        self.__build_model()
        optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        loss = BayesianNeuralNetwork.negative_loglikelihood
        metrics = ["mse", "mae"]
        self.compile_model(optimizer=optimizer, loss=loss, metrics=metrics)
        # Fit the model
        if self.validation_features is None and self.validation_targets is None:
            self.history = self.model.fit(
                x=self.training_data,
                y=self.target,
                epochs=self.epochs,
                batch_size=self.batch,
            )
        else:
            self.history = self.model.fit(
                x=self.training_data,
                y=self.target,
                epochs=self.epochs,
                batch_size=self.batch,
                validation_data=(self.validation_features, self.validation_targets),
            )
        return

    def plot_confidence_band(
        self,
        targets: np.array,
        x_axis: np.array = None,
        output_folder: Path = Path("./"),
    ) -> None:
        """
        Plots fitted line of prediction

        :param output_folder: location where the plot is saved
        """
        output_folder.mkdir(parents=True, exist_ok=True)

        if x_axis == None:
            x_axis = range(len(self.prediction_mean))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_axis, targets, label="Test data")
        ax.fill_between(
            x_axis,
            self.prediction_lower_95_CI,
            self.prediction_upper_95_CI,
            color="blue",
            alpha=0.1,
            label="95% CI",
        )
        ax.plot(x_axis, self.prediction_mean, label="Mean prediction")
        plt.title("Bnn probabilistic")
        plt.legend()
        ax.grid()
        figname = str(Path(output_folder, "confidence_band_bnn.png"))
        plt.savefig(figname)
        plt.close()
        return
