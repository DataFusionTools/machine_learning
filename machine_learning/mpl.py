from dataclasses import dataclass
from typing import List, Union
from pathlib import Path
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sklearn
import matplotlib.pylab as plt

from .neural_networks import NeuralNetwork


@dataclass
class MPL(NeuralNetwork):
    """
    Class of the NN object and defines the NN settings.

    :param nb_neurons: Number of neurons in each hidden layer
    """

    nb_neurons: Union[List, None, np.array] = None

    def __build_model(
        self, scaled_training_data, target, output_activation_function: str
    ) -> None:
        """
        Function that builds a NN model

        :param scaled_training_data: Scaled training data of the model
        :param target: Target data of the model
        :param output_activation_function: Activation function of the output layer of the NN
        """
        inputs = keras.Input(shape=(self.training_data.shape[1],))
        x = scaled_training_data(inputs)
        # hidden layers
        for i in range(self.nb_hidden_layers):
            x = layers.Dense(
                self.nb_neurons[i],
                activation=self.activation_fct.value,
                kernel_regularizer=l2(self.regularisation),
            )(x)
        # output layer
        outputs = layers.Dense(
            target.shape[1],
            activation=output_activation_function,
            kernel_regularizer=l2(self.regularisation),
        )(x)

        # NN model
        self.model = keras.Model(inputs, outputs)
        self.model.summary()
        return

    def __encode_target_data(self):
        """
        Method that encodes targed data for classification
        """
        self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(self.target)
        self.target_label = self.encoder.classes_  # classes
        target_encoded = self.encoder.transform(self.target)
        target_encoded = to_categorical(
            target_encoded, num_classes=len(self.target_label)
        )
        return target_encoded

    def __calculate_weights_for_classification(self):
        """
        Method that calculates weights in case of imbalanced data
        """
        class_weight = None
        if self.weights == "Auto":
            weights = sklearn.utils.class_weight.compute_class_weight(
                "balanced", np.unique(self.target), self.target
            )
            class_weight = dict(enumerate(weights))
        return class_weight

    def train_classification(self) -> None:
        """
        Method that trains a NN classification model.
        """
        target_encoded = self.__encode_target_data()
        class_weight = self.__calculate_weights_for_classification()
        scaled_training_data = self.rescale_training_data()
        self.__build_model(scaled_training_data, target_encoded, "softmax")
        self.compile_model(["accuracy"])
        # Fit the model
        self.history = self.model.fit(
            self.training_data,
            target_encoded,
            epochs=self.epochs,
            batch_size=self.batch,
            class_weight=class_weight,
        )
        scores = self.model.evaluate(self.training_data, target_encoded)
        # Evaluate
        print(
            f"{self.model.metrics_names[1]} of training: {round(scores[1] * 100, 2)} %"
        )
        self.accuracy = scores[1]
        return

    def train_regression(self) -> None:
        """
        Method that trains a NN regression model.
        """
        # scale training data
        scaled_training_data = self.rescale_training_data()
        self.__build_model(scaled_training_data, self.target, "linear")
        self.compile_model(["mse", "mae"])
        # Fit the model
        if self.validation_features is None and self.validation_targets is None:
            self.history = self.model.fit(
                self.training_data,
                self.target,
                epochs=self.epochs,
                batch_size=self.batch,
            )
        else:
            self.history = self.model.fit(
                self.training_data,
                self.target,
                epochs=self.epochs,
                batch_size=self.batch,
                validation_data=(self.validation_features, self.validation_targets),
            )
        # Evaluate
        scores = self.model.evaluate(self.training_data, self.target)
        print(f"{self.model.metrics_names[1]}: {round(scores[1], 2)}")
        return

    def plot_confusion(
        self, validation: np.ndarray, output_folder: Path = Path("./")
    ) -> None:
        """
        Plots the confusion matrix for the validation dataset

        :param validation: Validation data at the predicted points
        :param output_folder: location where the plot is saved
        """
        if not (self.classification):
            raise ReferenceError(
                "Method plot_confusion can only be used after performing classification process. "
            )
        directory = str(output_folder)
        if not os.path.isdir(directory):
            os.makedirs(directory)

        validation = np.array(list(validation))
        self.prediction = np.array(list(self.prediction))

        # compute confusion matrix
        confusion = confusion_matrix(
            self.encoder.transform(validation),
            self.encoder.transform(self.prediction),
            labels=self.encoder.transform(self.target_label),
        )  # , normalize="true")

        print(f"Confusion matrix:\n {confusion}")

        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion, display_labels=self.target_label
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_position([0.15, 0.15, 0.8, 0.8])
        disp.plot(cmap="binary", ax=ax)
        # disp.im_.set_clim(0, 1)
        directory = Path(
            output_folder,
            "confusion_matrix_epochsize%d_batchsize%d_regularization%d.png"
            % (self.epochs, self.batch, self.regularisation),
        )
        plt.savefig(directory)
        plt.close()

        return
