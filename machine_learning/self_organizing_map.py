from dataclasses import dataclass
from typing import List, Union
import numpy as np
import sompy
from collections import Counter
import pandas as pd

from .baseclass import BaseClassMachineLearning


@dataclass
class SOM(BaseClassMachineLearning):
    """
    Class of the Self Organizing Map.
    """

    mapsize: Union[List, None, np.array] = None
    mask: Union[List, None] = None
    mapshape: str = "planar"
    lattice: str = "rect"
    normalization: str = "var"
    initialization: str = "pca"
    neighborhood: str = "gaussian"
    training: str = "batch"
    name: str = "sompy"

    def train_som(self, data: np.ndarray, names: List) -> None:
        """
        Trains the SOM with the data
        """
        self.training_data = data
        self.component_names = names
        if self.mapsize is None:
            self.mapsize = [50, 60]
        som = sompy.SOMFactory.build(
            self.training_data,
            mapsize=self.mapsize,
            mask=self.mask,
            mapshape=self.mapshape,
            lattice=self.lattice,
            normalization=self.normalization,
            initialization=self.initialization,
            neighborhood=self.neighborhood,
            training=self.training,
            component_names=self.component_names,
            name="sompy",
        )

        # verbose='debug' will print detailed information
        som.train(n_job=1)  # , verbose='debug')

        # compute the umatrix
        u = sompy.umatrix.UMatrixView(
            50, 50, "umatrix", show_axis=True, text_size=8, show_text=True
        )

        # and its values
        UMAT = u.build_u_matrix(som, distance=1, row_normalized=False)

        # get the coordinates from the derived network (codebook)
        codebook = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)
        msz = som.codebook.mapsize
        cents = som.bmu_ind_to_xy(np.arange(0, msz[0] * msz[1]))

        # collect hits/node
        counts = Counter(som._bmu[0])
        counts = [counts.get(x, 0) for x in range(msz[0] * msz[1])]

        yv = cents[:, 0].astype(int)
        xv = cents[:, 1].astype(int)
        xyv = cents[:, 2].astype(int)

        # combine the results from the kohonen network with the original data
        df_cb = pd.DataFrame(data=codebook, columns=self.component_names)  # keywords)
        df_cb["U_matrix"] = UMAT.flatten("C")
        df_cb["X"] = xv
        df_cb["Y"] = yv
        df_cb["XY"] = xyv
        df_cb["Hits"] = counts
        self.codebook = df_cb
        self.bmu = som._bmu[0].astype(int)

    def train_classification(self, data, names) -> None:
        self.train_som(data=data, names=names)

    def train_regression(self, data, names) -> None:
        self.train_som(data=data, names=names)

    def train(self, data, names) -> None:
        self.train_som(data=data, names=names)

    def plot_umatrix_components(self):
        """
        Function to derive a interactive chart based on results of the Self-Organizing Map

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing the codebook and U-matrix. Derived from compute_som.

        Returns
        -------
        json
            Altair definition of interactive chart.
        """
        try:
            import altair as alt
        except ImportError as e:
            raise ImportError(
                f"{e}\nThe Python package Altair is required for this function."
            )

        if self.mapsize[0] <= 25:
            step_size = 12
        elif self.mapsize[0] < 50:
            step_size = 9
        else:
            step_size = 6

        if not hasattr(self, "codebook"):
            raise ValueError(f"Train SOM first")

        data_fields = self.codebook.columns.to_list()
        # data_fields = get_columns_from_csv_url(csv_url)
        data_fields.remove("X")
        data_fields.remove("Y")
        data_fields.remove("XY")
        data_fields.remove("U_matrix")

        input_dropdown = alt.binding_select(options=data_fields)
        if alt.__version__ > "4.3":
            selection = alt.selection_single(
                fields=["parameter"],
                bind=input_dropdown,
                name="selected",
                value=data_fields[0],  # valid after altair 5.0 is released
                # init={"parameter": data_fields[0]},  # depr in altair 5.0
            )
        else:
            selection = alt.selection_single(
                fields=["parameter"],
                bind=input_dropdown,
                name="selected",
                # value=data_fields[0], # valid after altair 5.0 is released
                init={"parameter": data_fields[0]},  # depr in altair 5.0
            )

        scale_color = alt.Scale(
            range=[
                "#3D3D3D",
                "#F0F8FF",
                "cornflowerblue",
                "mediumseagreen",
                "#FFEE00",
                "darkorange",
                "firebrick",
            ],
            zero=False,
            nice=False,
        )
        scale_color_umatrix = alt.Scale(
            range=[
                "#98438C",
                "#5D68AA",
                "#ACD424",
                "#F3DD0B",
                "#FBAD03",
                "#FB5605",
                "firebrick",
            ],
            zero=False,
            nice=False,
        )
        brush = alt.selection(type="interval", name="BRUSH")

        # prepare base_matrix
        source_matrix = alt.Chart(self.codebook)

        # prepare umatrix chart
        umatrix = (
            source_matrix.mark_rect()
            .encode(
                x=alt.X(
                    "X:N", scale=alt.Scale(paddingInner=0.02), axis=None, sort=None
                ),
                y=alt.Y(
                    "Y:N", scale=alt.Scale(paddingInner=0.02), axis=None, sort=None
                ),
                opacity=alt.condition(brush, alt.value(1), alt.value(0.2)),
                color=alt.Color(
                    "U_matrix:Q",
                    scale=scale_color_umatrix,
                    legend=alt.Legend(title="U-Matrix", labelLimit=300, orient="top"),
                ),
            )
            .add_selection(brush)
            .properties(width={"step": step_size}, height={"step": step_size})
        )

        # prepare parameter component chart
        components = (
            source_matrix.mark_rect()
            .encode(
                x=alt.X(
                    "X:N", scale=alt.Scale(paddingInner=0.02), axis=None, sort=None
                ),
                y=alt.Y(
                    "Y:N", scale=alt.Scale(paddingInner=0.02), axis=None, sort=None
                ),
                opacity=alt.condition(brush, alt.value(1), alt.value(0.2)),
                # cannot yet use expr for title. see https://github.com/vega/vega-lite/issues/7264
                color=alt.Color(
                    "value:Q",
                    scale=scale_color,
                    legend=alt.Legend(
                        title="selected_parameter", labelLimit=300, orient="top"
                    ),
                ),
            )
            .transform_fold(data_fields, as_=["parameter", "value"])
            .add_selection(selection)
            .add_selection(brush)
            .transform_filter(selection)
            .properties(width={"step": step_size}, height={"step": step_size})
        )

        cts_base = source_matrix.mark_bar(color="lightgray").encode(
            y=alt.Y("Hits:Q", aggregate="sum")
        )
        cts_slice = (
            source_matrix.mark_bar(color="black")
            .encode(y=alt.Y("Hits:Q", aggregate="sum", title="Selected Raw Records"))
            .transform_filter(brush)
        )

        comb = alt.hconcat(umatrix, components).resolve_scale(
            color="independent"
        ) | alt.layer(cts_base, cts_slice).properties(
            height=step_size * self.mapsize[0], width=40
        )

        # vl['hconcat'][1]['encoding']['color']['legend']['title'] = {'signal': 'select_parameter'}
        # vl['autosize'] = {'type':'fit', 'resize':True}
        # vl['background'] = 'transparent'

        return comb

    def plot_umatrix_components_target(self, df_target, target_var="Output"):

        df_target = df_target.copy()
        df_target.loc[:, "bmu"] = self.bmu

        try:
            import altair as alt
        except ImportError as e:
            raise ImportError(
                f"{e}\nThe Python package Altair is required for this function."
            )

        if self.mapsize[0] <= 25:
            step_size = 12
        elif self.mapsize[0] < 50:
            step_size = 9
        else:
            step_size = 6

        if not hasattr(self, "codebook"):
            raise ValueError(f"Train SOM first")

        if self.mapsize[0] <= 25:
            step_size = 12
        elif self.mapsize[0] < 50:
            step_size = 9
        else:
            step_size = 6

        data_fields = self.codebook.columns.to_list()
        data_fields.remove("X")
        data_fields.remove("Y")
        data_fields.remove("XY")
        # data_fields.remove("U_matrix")

        input_dropdown = alt.binding_select(options=data_fields)
        if alt.__version__ > "4.3":
            selection = alt.selection_single(
                fields=["parameter"],
                bind=input_dropdown,
                name="selected",
                value="U_matrix",  # valid after altair 5.0 is released
                # init={"parameter": data_fields[0]},  # depr in altair 5.0
            )
        else:
            selection = alt.selection_single(
                fields=["parameter"],
                bind=input_dropdown,
                name="selected",
                # value=data_fields[0], # valid after altair 5.0 is released
                init={"parameter": "U_matrix"},  # depr in altair 5.0
            )

        scale_color = alt.Scale(
            range=[
                "#3D3D3D",
                "#F0F8FF",
                "cornflowerblue",
                "mediumseagreen",
                "#FFEE00",
                "darkorange",
                "firebrick",
            ],
            zero=False,
            nice=False,
        )

        brush = alt.selection(type="interval", name="BRUSH")

        components = (
            alt.Chart(self.codebook)
            .mark_rect()
            .encode(
                x=alt.X(
                    "X:N", scale=alt.Scale(paddingInner=0.02), axis=None, sort=None
                ),
                y=alt.Y(
                    "Y:N", scale=alt.Scale(paddingInner=0.02), axis=None, sort=None
                ),
                opacity=alt.condition(brush, alt.value(1), alt.value(0.2)),
                # cannot yet use expr for title. see https://github.com/vega/vega-lite/issues/7264
                color=alt.Color(
                    "value:Q",
                    scale=scale_color,
                    legend=alt.Legend(
                        title="selected_parameter", labelLimit=300, orient="top"
                    ),
                ),
            )
            .transform_fold(data_fields, as_=["parameter", "value"])
            .add_selection(selection)
            .add_selection(brush)
            .transform_filter(selection)
            .properties(width={"step": step_size}, height={"step": step_size})
        )

        dynamic_target = (
            alt.Chart(df_target)
            .mark_bar()
            .encode(x=target_var, y="count()", color=target_var)
            .transform_lookup(
                lookup="bmu", from_=alt.LookupData(self.codebook, "XY"), as_="lu"
            )
            .transform_calculate("X", "datum.lu.X")
            .transform_calculate("Y", "datum.lu.Y")
            .transform_filter(brush)
        )

        # combine charts in layer and horizontal concat
        comb = alt.hconcat(components, dynamic_target)
        return comb
