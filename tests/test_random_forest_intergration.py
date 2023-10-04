from core.data_input import Data, Variable, Geometry
from core.utils import CreateInputsML, AggregateMethod
from machine_learning.random_forest import RandomForest
import machine_learning.enumeration_classes as class_enums

import tensorflow

import pytest
import numpy as np
from tests.utils import TestUtils
import pickle
import matplotlib.pyplot as plt
from pathlib import Path


class TestIntergrationTestRandomForest:
    @pytest.mark.intergrationtest
    def test_tutorial_non_flat_data(self):
        input_files = str(
            TestUtils.get_test_files_from_local_test_dir("", "test_case_DF.pickle")[0]
        )
        with open(input_files, "rb") as f:
            (cpts, resistivity, insar) = pickle.load(f)
        # create List[Data]
        cpts_list = []
        for name, item in cpts.items():
            location = Geometry(x=item["coordinates"][0], y=item["coordinates"][1], z=0)
            cpts_list.append(
                Data(
                    location=location,
                    independent_variable=Variable(value=item["NAP"], label="NAP"),
                    variables=[
                        Variable(value=item["water"], label="water"),
                        Variable(value=item["tip"], label="tip"),
                        Variable(value=item["IC"], label="IC"),
                        Variable(value=item["friction"], label="friction"),
                    ],
                )
            )

        resistivity_list = []
        for name, item in resistivity.items():
            location = Geometry(x=item["coordinates"][0], y=item["coordinates"][1], z=0)
            resistivity_list.append(
                Data(
                    location=location,
                    independent_variable=Variable(value=item["NAP"], label="NAP"),
                    variables=[
                        Variable(value=item["resistivity"], label="resistivity")
                    ],
                )
            )

        insar_list = []
        for counter, coordinates in enumerate(insar["coordinates"]):
            location = Geometry(x=coordinates[0], y=coordinates[1], z=0)
            insar_list.append(
                Data(
                    location=location,
                    independent_variable=Variable(value=insar["time"], label="time"),
                    variables=[
                        Variable(
                            value=insar["displacement"][counter], label="displacement"
                        )
                    ],
                )
            )

        create_features = CreateInputsML()
        aggregated_features = create_features.find_closer_points(
            input_data=cpts_list,
            combined_data=insar_list,
            aggregate_method=AggregateMethod.SUM,
            interpolate_on_independent_variable=False,
            aggregate_variable="displacement",
            number_of_points=2,
        )
        aggregated_features = create_features.find_closer_points(
            input_data=aggregated_features,
            combined_data=resistivity_list,
            aggregate_method=AggregateMethod.MAX,
            aggregate_variable="resistivity",
            interpolate_on_independent_variable=True,
            number_of_points=2,
        )

        assert len(aggregated_features) != 0
        for aggregated_feature in aggregated_features:
            create_features.add_features(
                aggregated_feature,
                ["tip", "displacement", "resistivity"],
                use_independent_variable=True,
                use_location_as_input=(True, True, False),
            )
            create_features.add_targets(aggregated_feature, ["IC"])

        create_features.split_train_test_data()
        training_data = create_features.get_features_train(flatten=False)
        target_data = create_features.get_targets_train(flatten=False)
        validation_training = create_features.get_features_test(flatten=False)
        validation_target = create_features.get_targets_test(flatten=False)
        features_names = create_features.get_feature_names()

        nn = RandomForest(
            classification=False,
            n_estimator=np.linspace(1, 2, 2),
            max_depth=np.linspace(1, 4, 4),
            feature_names=features_names,
        )
        # Train with flat shape
        nn.train(
            training_data,
            target_data.flatten(),
        )
        nn.predict(validation_training)

        Path("./tests/test_output/test_intergration_rf").mkdir(
            parents=True, exist_ok=True
        )
        nn.plot_feature_importance(
            validation_training, Path("./tests/test_output/test_intergration_rf")
        )
        nn.plot_feature_importance_with_interaction_values(
            validation_training, Path("./tests/test_output/test_intergration_rf")
        )
        nn.plot_fitted_line(
            validation_target, Path("./tests/test_output/test_intergration_rf")
        )

        plt.clf()
        plt.plot(nn.prediction, validation_training.T[1], "ro")
        plt.plot(validation_target, validation_training.T[1], "bo")
        # plot 1/1 line
        plt.xlabel("IC")
        plt.ylabel("depth")
        plt.savefig("./tests/test_output/test_intergration_rf/comparison_depth.png")
