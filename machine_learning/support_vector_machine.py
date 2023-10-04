from dataclasses import dataclass
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVR
import numpy as np
import os

from .enumeration_classes import ActivationFunctionSVM, GammaList
from .baseclass import BaseClassMachineLearning



@dataclass
class SVM(BaseClassMachineLearning):
    """
    Class of the Support Vector Machine.

    :param kernel: Kernel type to be used in the algorithm
    :param gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
    """

    kernel: ActivationFunctionSVM = ActivationFunctionSVM.rbf
    gamma: GammaList = GammaList.scale

    def train_svm(self) -> None:
        """
        Trains the SVM with the data and multiple class target values
        """
        # labels target data
        self.target_label = list(set(self.target))
        # SVM model
        self.model = SVR(kernel=self.kernel.value, gamma=self.gamma.value)
        # Fit to data
        self.model.fit(self.training_data, self.target)
        # compute accuracy
        self.check_score()

    def train_classification(self) -> None:
        self.train_svm()

    def train_regression(self) -> None:
        self.train_svm()

    def check_score(self) -> None:
        """
        Computes score of training
        """
        # compute prediction of the training data
        target_predict = self.model.predict(self.training_data)

        if self.classification:
            self.accuracy = len(np.where(target_predict == self.target)[0]) / len(
                self.target
            )
            print(f"Accuracy of training: {round(self.accuracy * 100, 2)} %")
        else:
            self.accuracy = np.sqrt(((target_predict - self.target) ** 2).mean())
            print(f"RMSE of training: {round(self.accuracy, 2)}")
        return

    def predict(self, data: np.ndarray) -> None:
        """
        Predict the values at the data points

        :param data: dataset with features for prediction
        """
        self.prediction = self.model.predict(data)
        return

    def plot_confusion(self, validation: np.ndarray, output_folder: str = "./") -> None:
        """
        Plots the confusion matrix for the validation dataset

        :param validation: Validation data at the predicted points
        :param output_folder: location where the plot is saved
        """
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        confusion = confusion_matrix(
            validation, self.prediction, labels=self.target_label
        )  # , normalize="true")

        print(f"Confusion matrix:\n {confusion}")

        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion, display_labels=self.target_label
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_position([0.15, 0.15, 0.8, 0.8])
        disp.plot(cmap="binary", ax=ax)
        # disp.im_.set_clim(0, 1)
        plt.savefig(os.path.join(output_folder, "confusion_matrix"))
        plt.close()

        return
