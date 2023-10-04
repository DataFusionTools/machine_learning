from dataclasses import dataclass, field
from typing import List
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pylab as plt

from .neural_networks import NeuralNetwork


@dataclass
class Convolutional(NeuralNetwork):
    nb_filters: List[int] = field(default_factory=lambda: [64])
    length_filters: List[int] = field(default_factory=lambda: [3])
    n_dim: int = 1
    strides: int = 1

    def __encode_target_data(self):
        """
        Method that encodes targed data for classification
        """
        self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(np.append(self.target, "unknown"))
        self.target_label = self.encoder.classes_  # classes
        target_encoded = self.encoder.transform(self.target)
        target_encoded = to_categorical(
            target_encoded, num_classes=len(self.target_label)
        )
        return target_encoded

    def __build_model(
        self, scaled_training_data, target, output_activation_function: str
    ) -> None:
        """
        Function that builds a NN model

        :param scaled_training_data: Scaled training data of the model
        :param target: Target data of the model
        :param output_activation_function: Activation function of the output layer of the NN
        """
        # Build the model
        inputs = keras.Input(
            shape=self.training_data.reshape(
                self.training_data.shape[0], self.training_data.shape[1], self.n_dim
            ).shape[1:]
        )
        x = scaled_training_data(inputs)
        # hidden layers
        for i in range(self.nb_hidden_layers):
            x = layers.Conv1D(
                filters=self.nb_filters[i],
                kernel_size=self.length_filters[i],
                strides=self.strides,
                activation=self.activation_fct.value,
            )(x)
        # output layer
        x = layers.Flatten()(x)
        outputs = layers.Dense(
            target.shape[1],
            activation=output_activation_function,
            kernel_regularizer=l2(self.regularisation),
        )(x)
        # NN model
        self.model = keras.Model(inputs, outputs)
        self.model.summary()
        return

    def train_classification(self) -> None:
        target_encoded = self.__encode_target_data()
        scaled_training_data = self.rescale_training_data()
        self.__build_model(scaled_training_data, target_encoded, "softmax")
        self.compile_model(["accuracy"])

        # Fit the model
        self.history = self.model.fit(
            self.training_data,
            target_encoded,
            epochs=self.epochs,
            batch_size=self.batch,
        )

        scores = self.model.evaluate(self.training_data, target_encoded)
        print(
            f"{self.model.metrics_names[1]} of training: {round(scores[1] * 100, 2)} %"
        )
        self.accuracy = scores[1]

        return

    def train_regression(self) -> None:
        scaled_training_data = self.rescale_training_data()
        # normalise inputs
        self.__build_model(scaled_training_data, self.target, "softmax")
        self.compile_model(["mae", "mse"])
        # Fit the model
        self.history = self.model.fit(
            self.training_data,
            self.target,
            epochs=self.epochs,
            batch_size=self.batch,
        )
        scores = self.model.evaluate(self.training_data, self.target)
        print(f"{self.model.metrics_names[1]}: {round(scores[1], 2)}")

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
        output_folder.mkdir(parents=True, exist_ok=True)

        new_data_list = list(validation)
        for unique_item in np.unique(validation):
            if unique_item not in self.encoder.classes_:
                new_data_list = [
                    "unknown" if x == unique_item else x for x in new_data_list
                ]
        validation = np.array(new_data_list)

        new_data_list = list(self.prediction)
        for unique_item in np.unique(self.prediction):
            if unique_item not in self.encoder.classes_:
                new_data_list = [
                    "unknown" if x == unique_item else x for x in new_data_list
                ]
        self.prediction = np.array(new_data_list)

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
        figname = str(
            Path(
                output_folder,
                "confusion_matrix_epochsize%d_batchsize%d_regularization%d.png"
                % (self.epochs, self.batch, self.regularisation),
            )
        )
        plt.savefig(figname)
        plt.close()
        return
