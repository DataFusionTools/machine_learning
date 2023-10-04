from machine_learning.support_vector_machine import SVM
import pytest
import numpy as np
from sklearn import preprocessing
from utils import TestUtils

np.random.seed(1)


class TestSVM:
    @pytest.mark.intergrationtest
    def test_SVM_classification(self):
        # read data from file
        input_files = str(
            TestUtils.get_test_files_from_local_test_dir("", "NN_data.csv")[0]
        )
        with open(input_files, "r") as f:
            data = f.read().splitlines()
        data = [i.split(";") for i in data[1:100]]
        features = np.array(data)[:, :-1].astype(float)
        target = np.array(data)[:, -1]
        le = preprocessing.LabelEncoder()
        target = le.fit_transform(target)
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
        rf = SVM(classification=True)
        rf.train(training_data, target_data)
        rf.predict(validation_training)

    @pytest.mark.intergrationtest
    def test_SVM_regression(self):
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
        # RF
        rf = SVM(classification=False)
        rf.train(training_data.astype(int), target_data.astype(int))
        assert rf.accuracy
        rf.predict(validation_training)
