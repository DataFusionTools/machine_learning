from core.data_input import Data, Variable, Geometry
from core.utils import CreateInputsML, AggregateMethod
from machine_learning.mpl import MPL
import machine_learning.enumeration_classes as class_enums

import pytest
from utils import TestUtils
import pickle
import matplotlib.pyplot as plt
from pathlib import Path


class TestTutorial:
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
        extra_validation_training = create_features.get_features_validation(
            flatten=False
        )
        extra_validation_target = create_features.get_targets_validation(flatten=False)
        feature_names = create_features.get_feature_names()

        nn = MPL(
            classification=False,
            nb_hidden_layers=3,
            nb_neurons=[18, 18, 14],
            activation_fct=class_enums.ActivationFunctions.relu,
            optimizer=class_enums.Optimizer.Adam,
            loss=class_enums.LossFunctions.mean_squared_error,
            epochs=1,
            batch=1,
            regularisation=0,
            feature_names=feature_names,
            validation_features=extra_validation_training,
            validation_targets=extra_validation_target,
        )
        # Train with flat shape
        nn.train(
            training_data,
            target_data,
        )
        nn.plot_feature_importance(validation_training, Path("./tests/test_output"))
        nn.plot_cost_function(output_folder=Path("./tests/test_output"))
        nn.predict(validation_training)
        nn.plot_fitted_line(validation_target, Path("./tests/test_output"))

        plt.clf()
        plt.plot(nn.prediction[:50], validation_training.T[1][:50], "ro")
        plt.plot(validation_target[:50], validation_training.T[1][:50], "bo")
        # plot 1/1 line
        plt.xlabel("IC")
        plt.ylabel("depth")
        plt.savefig("./tests/test_output/comparison_depth.png")
