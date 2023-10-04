from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
import shap

from .baseclass import BaseClassMachineLearning



@dataclass
class RandomForest(BaseClassMachineLearning):
    accuracy: Union[List, None, np.ndarray] = None
    encoder: Union[List, None, np.ndarray] = None
    model: Union[List, None, np.ndarray] = None
    n_estimator: np.ndarray = field(default_factory=lambda: np.linspace(1, 30, 30))
    max_depth: np.ndarray = field(default_factory=lambda: np.linspace(1, 20, 20))
    feature_names: Union[List, None] = None

    def __run_random_forest(self, estimator):
        self.n_estimator = self.n_estimator.astype(int)
        self.max_depth = self.max_depth.astype(int)
        self.target = self.target.ravel()

        if self.classification:
            # labels target data
            self.target_label = list(set(self.target))
        # RF model
        parameters = {"n_estimators": self.n_estimator, "max_depth": self.max_depth}
        self.model = GridSearchCV(estimator=estimator, param_grid=parameters)
        # Fit to data
        self.model.fit(self.training_data, self.target)
        # print parameters
        print(self.model.best_params_)

        # compute accuracy
        self.check_score()

    def train_classification(self):
        self.__run_random_forest(RandomForestClassifier())

    def train_regression(self):
        self.__run_random_forest(RandomForestRegressor())

    def check_score(self):
        """
        Computes score of training
        """
        # compute prediction of the training data
        target_predict = self.model.predict(self.training_data)
        if self.classification:
            # compute accuracy
            self.accuracy = len(np.where(target_predict == self.target)[0]) / len(
                self.target
            )
            print(f"Accuracy of training: {round(self.accuracy * 100, 2)} %")
        else:
            # compute accuracy: RMSE
            self.accuracy = np.sqrt(((target_predict - self.target) ** 2).mean())
            print(f"RMSE of training: {round(self.accuracy, 2)}")
        return

    def predict(self, data: np.ndarray) -> None:
        """
        Predict the values at the data points

        :param data: dataset with features for prediction
        """
        self.prediction = self.model.predict(data)

    def plot_confusion(
        self, validation: np.ndarray, output_folder: Path = Path("./")
    ) -> None:
        """
        Plots the confusion matrix for the validation dataset

        :param validation: Validation data at the predicted points
        :param output_folder: location where the plot is saved
        """
        output_folder.mkdir(parents=True, exist_ok=True)

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
        figname = str(Path(output_folder, "confusion_matrix.png"))
        plt.savefig(figname)
        plt.close()

        return

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
        f = plt.figure()
        explainer = shap.TreeExplainer(self.model.best_estimator_)
        shap_values = explainer.shap_values(input_data)
        shap.summary_plot(
            shap_values,
            input_data,
            feature_names=np.array(self.feature_names),
            show=False,
        )
        figname = str(Path(output_folder, "feature_importance.png"))
        f.savefig(figname, bbox_inches="tight", dpi=600)
        plt.close()
        return

    # TODO why is this plot not showing
    def plot_feature_importance_with_interaction_values(
        self, input_data: np.array, output_folder: Path = Path("./")
    ):
        """
        Function that plots the feature importance charts.
        This is done be using the shap package which uses Shapley values to explain machine learning models.
        For more information look in shap's package `website <https://shap.readthedocs.io/en/latest/index.html>`__

        :param output_folder: location where the plot is saved
        :param input_data: data to be used for the determination of the Shapley values.
        """
        f = plt.gcf()
        explainer = shap.TreeExplainer(self.model.best_estimator_)
        shap_interaction_values = explainer.shap_interaction_values(input_data)
        shap.summary_plot(
            shap_interaction_values,
            input_data,
            feature_names=np.array(self.feature_names),
            show=False,
        )
        figname = str(
            Path(output_folder, "feature_importance_with_interaction_values.png")
        )
        f.savefig(figname)
        plt.close()
        return
