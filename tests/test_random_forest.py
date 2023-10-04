from pathlib import Path
from machine_learning.random_forest import RandomForest
import machine_learning.enumeration_classes as enum_classes
import pytest
import numpy as np
from tests.utils import TestUtils

np.random.seed(1)


class TestRandomForest:
    @pytest.mark.intergrationtest
    def test_RandomForest_classification(self):
        # read data from file
        input_files = str(
            TestUtils.get_test_files_from_local_test_dir("", "NN_data.csv")[0]
        )
        with open(input_files, "r") as f:
            data = f.read().splitlines()
        data = [i.split(";") for i in data[1:15]]
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
        # RF
        rf = RandomForest(classification=True)
        rf.train(training_data.astype(int), target_data)
        rf.predict(validation_training.astype(int))
        rf.plot_confusion(
            validation_target, output_folder=Path("tests/test_output/example_rf")
        )
        assert Path("tests/test_output/example_rf/confusion_matrix.png").is_file()

    @pytest.mark.intergrationtest
    def test_RandomForest_regression(self):
        # read data from file
        input_files = str(
            TestUtils.get_test_files_from_local_test_dir("", "NN_data.csv")[0]
        )
        with open(input_files, "r") as f:
            data = f.read().splitlines()
        data = [i.split(";") for i in data[1:10]]
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
        # RF
        rf = RandomForest(classification=False)
        rf.train(training_data.astype(int), target_data.astype(int))
        assert rf.accuracy
        rf.predict(validation_training)
