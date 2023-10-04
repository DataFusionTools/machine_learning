from dataclasses import dataclass
from typing import List, Union
from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from core.base_class import BaseClass


@dataclass
class BaseClassMachineLearning(BaseClass):
    classification: bool
    training_data: Union[List, None, np.array] = None
    target: Union[List, None, np.array] = None
    target_label: Union[List, None, np.array] = None
    prediction: Union[List, None, np.array] = None

    @abstractmethod
    def train_classification(self):
        raise NotImplementedError(
            "The method should be implemented in concrete classes."
        )

    @abstractmethod
    def train_regression(self):
        raise NotImplementedError(
            "The method should be implemented in concrete classes."
        )

    def train(self, data: np.ndarray, target: np.ndarray) -> None:
        """
        Trains the NN with the data and multiple class target values
        based on the model selected (classification or regression).

        :param data: data features
        :param target: multiple class target value
        """
        # variables
        self.training_data = data
        self.target = target
        if self.classification:
            self.train_classification()
        else:
            self.train_regression()
        return

    @abstractmethod
    def predict(self):
        raise NotImplementedError(
            "The method should be implemented in concrete classes."
        )

    def plot_fitted_line(
        self, validation_target: np.ndarray, output_folder: Path = Path("./")
    ) -> None:
        """
        Plots fitted line of prediction

        :param output_folder: location where the plot is saved
        """
        output_folder.mkdir(parents=True, exist_ok=True)

        # max and mins
        max_all = max(
            np.amax(self.prediction.flatten()), np.amax(validation_target.flatten())
        )
        min_all = min(
            np.amin(self.prediction.flatten()), np.amin(validation_target.flatten())
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_position([0.15, 0.15, 0.8, 0.8])
        ax.plot(self.prediction.flatten(), validation_target.flatten(), "ro")
        ax.plot([min_all - 0.2, max_all + 0.2], [min_all - 0.2, max_all + 0.2], "-")
        ax.set_xlim([min_all - 0.1, max_all + 0.1])
        ax.set_ylim([min_all - 0.1, max_all + 0.1])
        ax.set_xlabel("prediction")
        ax.set_ylabel("actual data")
        ax.grid()
        figname = str(Path(output_folder, "fitted_line.png"))
        plt.savefig(figname)
        plt.close()
        return None
