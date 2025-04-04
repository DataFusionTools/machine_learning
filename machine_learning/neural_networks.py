from dataclasses import dataclass
from typing import List, Union
from tensorflow.keras.losses import KLDivergence
from tensorflow.keras import layers
from tensorflow.keras.layers import Rescaling
import matplotlib.pylab as plt
import numpy as np
from pathlib import Path

import shap

from .baseclass import BaseClassMachineLearning

from .enumeration_classes import (
    Optimizer,
    ActivationFunctions,
    LossFunctions,
)


@dataclass
class NeuralNetwork(BaseClassMachineLearning):
    """
    Initialises the NN object and defines the NN settings.
    NN defined for regression problems.

    :param nb_hidden_layers: number of hidden layers
    :param activation_fct: (optional: default 'sigmoid') Type of activation function
    :param optimizer: (optional: default 'Adam') Type of minimisation
    :param loss: (optional: default 'binary_crossentropy') Type of activation loss function
    :param epochs: (optional: default 500) Number of epochs
    :param batch: (optional: default 32) Number of batch in each epoch realisation
    :param regularisation: (optional: default 0) Factor for regularisation
    :param weights: (optional: default None) Weights for the categories
    """

    history: Union[List, None, np.ndarray] = None
    encoder_features: Union[List, None, np.ndarray] = None
    encoder_target: Union[List, None, np.ndarray] = None
    prediction: Union[List, None, np.ndarray] = None
    model: Union[List, None, np.ndarray] = None
    kl: Union[List, None, np.ndarray] = None
    weights: Union[List, None] = None
    nb_hidden_layers: int = 1
    activation_fct: ActivationFunctions = ActivationFunctions.sigmoid
    optimizer: Optimizer = Optimizer.Adam
    loss: LossFunctions = LossFunctions.mean_absolute_error
    epochs: int = 500
    batch: int = 32
    regularisation: int = 0
    feature_names: Union[List, None] = None
    validation_targets: Union[List, None, np.ndarray] = None
    validation_features: Union[List, None, np.ndarray] = None
    probabilistic: bool = False

    def plot_cost_function(self, output_folder: Path = Path("./")) -> None:
        """
        Plots the cost function

        :param output_folder: location where the plot is saved
        """
        output_folder.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_position([0.15, 0.15, 0.8, 0.8])
        ax.plot(self.history.history["loss"], label="Loss training")
        if self.history.history.get("val_loss", False):
            ax.plot(self.history.history["val_loss"], label="Loss validation")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid()
        figname = str(Path(output_folder, "performance.png"))
        plt.savefig(figname)
        plt.close()

    def plot_feature_importance(
        self, input_data: np.array, output_folder: Path = Path("./")
    ):
        """
        Function that plots the feature importance charts.
        This is done be using the shap package which uses Shapley values to explain machine learning models.
        For more information look in shap's package `website <https://shap.readthedocs.io/en/latest/index.html>`__

        :param output_folder: location where the plot is saved
        :param input_data: data to be used for the determination of the Shapley values.
        """
        # tensorflow.compat.v1.disable_v2_behavior()  # This is required for the shap plotter

        # if output folder does not exist creates it
        output_folder.mkdir(parents=True, exist_ok=True)

        f = plt.figure()
        background = self.training_data[
            np.random.choice(self.training_data.shape[0], 100, replace=False)
        ]
        explainer = shap.KernelExplainer(self.model, background)
        shap_values = explainer.shap_values(input_data)
        shap.summary_plot(
            shap_values[0], input_data, feature_names=self.feature_names, show=False
        )
        figname = str(Path(output_folder, "feature_importance.png"))
        f.savefig(figname, bbox_inches="tight", dpi=600)
        plt.close()
        return

    def kl_div(self, y_true, y_pred):
        """
        Compute KL divergence

        :param y_true: true value
        :param y_pred: predicted value with NN
        """
        if self.classification:
            raise ReferenceError("Method kl_div can only be for regression ")
        kl = KLDivergence()
        self.kl = kl(y_true, y_pred).numpy()
        return

    def rescale_training_data(self):
        """
        Rescaling process of training data
        """
        return Rescaling(
            1 / np.max(self.training_data, axis=0)
        )

    def compile_model(self, metrics: List[str], loss=None, optimizer=None):
        """
        Method that compiles NN with particular metrics

        :param metrics: List of metrics to be used during training
        """
        if loss is None:
            loss = self.loss.value
        if optimizer is None:
            optimizer = self.optimizer.value
        # Compile model
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def predict(self, data: np.ndarray) -> None:
        """
        Predict the values at the data points with trained NN

        :param data: dataset with features for prediction
        """

        prediction_encoded = self.model.predict(data)
        if self.probabilistic:
            self.prediction = prediction_encoded
            prediction_distribution = self.model(data)
            self.prediction_mean = prediction_distribution.mean().numpy().flatten()
            self.prediction_stdv = prediction_distribution.stddev().numpy().flatten()

            # The 95% CI is computed as mean ± (1.96 * stdv)
            self.prediction_upper_95_CI = (
                self.prediction_mean + (1.96 * self.prediction_stdv)
            ).tolist()
            self.prediction_lower_95_CI = (
                self.prediction_mean - (1.96 * self.prediction_stdv)
            ).tolist()
        else:
            if self.classification:
                self.prediction = np.array(
                    [
                        self.target_label[i]
                        for i in np.argmax(prediction_encoded, axis=1)
                    ]
                )
            else:
                self.prediction = prediction_encoded
        return
