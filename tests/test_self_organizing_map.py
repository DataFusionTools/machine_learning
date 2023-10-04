from machine_learning.self_organizing_map import SOM
import pytest
import numpy as np
from tests.utils import TestUtils
import pandas as pd


class TestSOM:
    @pytest.mark.intergrationtest
    def test_SOM_training(self):
        # read data from file
        input_files = str(
            TestUtils.get_test_files_from_local_test_dir("", "NN_data.csv")[0]
        )
        with open(input_files, "r") as f:
            data = f.read().splitlines()
        # prepare features
        component_names = [i.split(";") for i in data[0:1]][0]
        training_data = [i.split(";") for i in data[1:15]]

        features = np.array(training_data)[:, :-1].astype(float)
        feature_names = component_names[:-1]

        target = np.array(training_data)[:, -1]
        target_name = component_names[-1]

        # test trained SOM
        som = SOM(classification=False, mapsize=[5, 5])
        som.train(data=features, names=feature_names)

        assert som.codebook.shape == (25, 13)
        assert 25 < som.codebook.U_matrix.sum() < 30

        # test visz
        umat_comp = som.plot_umatrix_components()
        df_target = pd.DataFrame(target, columns=[target_name])
        umat_comp_target = som.plot_umatrix_components_target(df_target, target_name)

        assert "$schema" in umat_comp.to_dict()
        assert "$schema" in umat_comp_target.to_dict()
