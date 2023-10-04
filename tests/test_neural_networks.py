from pathlib import Path
from machine_learning.mpl import MPL
import machine_learning.enumeration_classes as enum_classes
import pytest
import numpy as np
from utils import TestUtils

np.random.seed(1)


class TestMPL:
    @pytest.mark.intergrationtest
    def test_MPL_classification(self):
        # read data from file
        input_files = str(
            TestUtils.get_test_files_from_local_test_dir("", "NN_data.csv")[0]
        )
        with open(input_files, "r") as f:
            data = f.read().splitlines()
        data = [i.split(";") for i in data[1:]]
        features = np.array(data)[:, :-1].astype(float)
        target = np.array(data)[:, -1]
        # percentage
        percent = 0.8
        # number of data for training
        nb_points = int(len(data) * percent)
        # indexes for training
        idx_tra = np.random.choice(len(data), size=nb_points, replace=False)
        # indexes for validation
        idx_val = list(set(range(len(data))) - set(idx_tra))

        # Training and validation data
        training_data = features[idx_tra]
        target_data = target[idx_tra]

        validation_training = features[idx_val]
        validation_target = target[idx_val]
        # Set MPL
        nn = MPL(
            classification=True,
            nb_hidden_layers=2,
            nb_neurons=[8, 4],
            activation_fct=enum_classes.ActivationFunctions.sigmoid,
            optimizer=enum_classes.Optimizer.Adam,
            loss=enum_classes.LossFunctions.binary_crossentropy,
            epochs=3,
            batch=32,
            regularisation=0,
        )
        assert nn
        nn.train(training_data, target_data)
        assert nn.model
        nn.predict(validation_training)
        assert nn.accuracy
        nn.plot_cost_function(
            output_folder=Path("tests/test_output/test_MPL_classification")
        )
        assert Path(
            "tests",
            "test_output",
            "test_MPL_classification",
            "performance.png",
        ).is_file()
        nn.plot_confusion(
            validation_target,
            output_folder=Path("./tests/test_output/test_MPL_classification"),
        )
        assert Path(
            "tests",
            "test_output",
            "test_MPL_classification",
            "confusion_matrix_epochsize3_batchsize32_regularization0.png",
        ).is_file()

    @pytest.mark.intergrationtest
    def test_MPL_regression(self):
        # read data from file
        input_files = str(
            TestUtils.get_test_files_from_local_test_dir("", "NN_data.csv")[0]
        )
        with open(input_files, "r") as f:
            data = f.read().splitlines()
        data = [i.split(";") for i in data[1:]]
        features = np.array(data)[:, :-2].astype(float)
        target = np.array(data)[:, -2].astype(float)
        # percentage
        percent = 0.8
        # number of data for training
        nb_points = int(len(data) * percent)
        # indexes for training
        idx_tra = np.random.choice(len(data), size=nb_points, replace=False)
        # indexes for validation
        idx_val = list(set(range(len(data))) - set(idx_tra))

        # Training and validation data
        training_data = features[idx_tra]
        target_data = target[idx_tra]
        validation_training = features[idx_val]
        validation_target = target[idx_val]
        # 1D Convolutional NN
        nn = MPL(
            classification=False,
            nb_hidden_layers=2,
            nb_neurons=[8, 4],
            activation_fct=enum_classes.ActivationFunctions.sigmoid,
            optimizer=enum_classes.Optimizer.Adam,
            loss=enum_classes.LossFunctions.mean_absolute_error,
            epochs=2,
            batch=32,
            regularisation=0,
        )
        assert nn
        nn.train(training_data, np.reshape(target_data, (target_data.shape[0], 1)))
        assert nn.model
        nn.predict(validation_training)
        nn.plot_cost_function(
            output_folder=Path("./tests/test_output/test_MPL_regression")
        )
        assert Path(
            "tests", "test_output", "test_MPL_regression", "performance.png"
        ).is_file()
        nn.kl_div(
            y_true=np.reshape(validation_target, (validation_target.shape[0], 1)),
            y_pred=nn.prediction,
        )


    @pytest.mark.intergrationtest
    def test_MPL_regression_two_outputs(self):
        # read data from file
        input_files = str(
            TestUtils.get_test_files_from_local_test_dir("", "NN_data.csv")[0]
        )
        with open(input_files, "r") as f:
            data = f.read().splitlines()
        data = [i.split(";") for i in data[1:]]
        features = np.array(data)[:, :-2].astype(float)
        target = np.array(data)[:, -2].astype(float)
        # percentage
        percent = 0.8
        # number of data for training
        nb_points = int(len(data) * percent)
        # indexes for training
        idx_tra = np.random.choice(len(data), size=nb_points, replace=False)
        # indexes for validation
        idx_val = list(set(range(len(data))) - set(idx_tra))

        # Training and validation data
        training_data = features[idx_tra]
        target_data = target[idx_tra]
        validation_training = features[idx_val]
        validation_target = target[idx_val]
        # 1D Convolutional NN
        nn = MPL(
            classification=False,
            nb_hidden_layers=2,
            nb_neurons=[8, 4],
            activation_fct=enum_classes.ActivationFunctions.sigmoid,
            optimizer=enum_classes.Optimizer.Adam,
            loss=enum_classes.LossFunctions.mean_absolute_error,
            epochs=2,
            batch=32,
            regularisation=0,
        )
        assert nn
        target_data = np.array([target_data, target_data]).T
        nn.train(training_data, target_data)
        assert nn.model
        nn.predict(validation_training)
        nn.plot_cost_function(
            output_folder=Path("./tests/test_output/test_MPL_regression")
        )
        assert Path(
            "tests", "test_output", "test_MPL_regression", "performance.png"
        ).is_file()
        nn.kl_div(
            y_true=np.reshape(validation_target, (validation_target.shape[0], 1)),
            y_pred=nn.prediction,
        )