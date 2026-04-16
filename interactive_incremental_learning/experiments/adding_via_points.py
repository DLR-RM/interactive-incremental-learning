#!/usr/bin/env python

################################################################################
# Copyright (c) 2025. Markus Knauer, Joao Silverio                             #
# Licensed under the MIT License. See LICENSE file for details.                 #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 04-03-2025                                                             #
# Author: Markus Knauer                                                        #
# E-mail: markus.knauer@dlr.de                                                 #
# Website: https://github.com/DLR-RM/interactive-incremental-learning          #
################################################################################

"""
Script to run the experiment to add via-points

If you are using this code please cite us:
M. Knauer, A. Albu-Schäffer, F. Stulp and J. Silvério, "Interactive Incremental Learning of Generalizable
Skills With Local Trajectory Modulation," in IEEE Robotics and Automation Letters (RA-L), vol. 10, no. 4,
pp. 3398-3405, April 2025, doi: 10.1109/LRA.2025.3542209

See CITATION.bib for the bib file!
"""

__author__ = "Markus Knauer, Joao Silverio"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from interactive_incremental_learning import ConfigParams
from interactive_incremental_learning.common.data_object import (
    ORIENTATION_QUAT_W_FIRST,
    POSITION,
    DataObject,
    DataObjectType,
    InputType,
)
from interactive_incremental_learning.common.tp_kmp import TPKMP, TPKMPConfigParams


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the via-point insertion experiment."""

    end_time_stamp: float = 1.0

    add_new_frame_in_between: bool = True

    frames_data = np.array([[0.3294, 0.5512, 0.1555, 1.0, 0.0, 0.0, 0.0], [-0.107, -0.6954, 0.0672, 0.707, 0.0, 0.0, -0.707]])


class AddViaPointsExperiment:
    """Experiment demonstrating incremental learning by adding via-point trajectory constraints."""

    def run(
        self,
        tp_kmp_model: TPKMP,
        tp_kmp_params: TPKMPConfigParams,
        frames_used_for_demonstrations: list[DataObject],
        params: ConfigParams,
        plot: bool = False,
    ):
        """Adds via-points at t=0.80-0.82 to constrain the trajectory endpoint and re-predicts.

        :param tp_kmp_model: trained TP-KMP model
        :param tp_kmp_params: model hyperparameters
        :param frames_used_for_demonstrations: frames from the training demonstrations
        :param params: data structure configuration
        :param plot: if True, displays and saves the figure
        """
        ######## TP-KMP Prediction for time horizon ########
        experiment_config = ExperimentConfig()
        x_test = np.linspace(0, experiment_config.end_time_stamp, tp_kmp_params.number_of_test_points)

        # Calculate prediction before adding via-points
        _, _, _, _ = tp_kmp_model.predict(x_test=x_test)

        ### Add via-points ###
        # This is temporary to add a via-point to the last frame manually (29022024)
        via_point_position_global_one = [-0.13, -0.678, 0.064, 0.0103, -0.3385, -0.9408, -0.0136]
        via_point_list_global = []
        via_point_list_global.append(
            DataObject(
                data=np.array([[0.80, *via_point_position_global_one, 1]]),
                data_structure=[InputType.T, *POSITION, *ORIENTATION_QUAT_W_FIRST, InputType.FRAME_INDEX],
                data_type=DataObjectType.ARRAY_2D,
            )
        )
        via_point_list_global.append(
            DataObject(
                data=np.array([[0.81, *via_point_position_global_one, 1]]),
                data_structure=[InputType.T, *POSITION, *ORIENTATION_QUAT_W_FIRST, InputType.FRAME_INDEX],
                data_type=DataObjectType.ARRAY_2D,
            )
        )
        via_point_list_global.append(
            DataObject(
                data=np.array([[0.82, *via_point_position_global_one, 1]]),
                data_structure=[InputType.T, *POSITION, *ORIENTATION_QUAT_W_FIRST, InputType.FRAME_INDEX],
                data_type=DataObjectType.ARRAY_2D,
            )
        )

        via_local_to_frame2 = []
        for via_point in via_point_list_global:
            # Get the first frame from frames_used_for_demonstrations[1] for frame 1
            frame_1 = frames_used_for_demonstrations[1].get_rows([0])
            via_local_to_frame2.append(via_point.map_to_different_data_object(frame_1))
        tp_kmp_model.add_via_points_locally_to_kmp(
            kmp_idx=1, time_stamps_for_insertions=[0.8, 0.81, 0.82], via_local=via_local_to_frame2, variance=1e-8
        )
        # For plotting
        for via_point_obj in via_point_list_global:
            tp_kmp_model.global_via_points.append(via_point_obj)

        # Recalculate prediction
        mu_pred_local, sigma_pred_local, sigma_pred_epistemic_local, sigma_pred_aleatoric_local = tp_kmp_model.predict(
            x_test=x_test
        )

        # Set new object positions for trajectory prediction
        if experiment_config.frames_data is None:
            pred_frames_data = frames_used_for_demonstrations[0].get_array(
                frames_used_for_demonstrations[0].get_data_structure()
            )
        else:
            pred_frames_data = experiment_config.frames_data
        pred_frames_obj = DataObject(
            data=pred_frames_data, data_structure=params.frames_data_structure, data_type=DataObjectType.ARRAY_2D
        )

        # Combine local predictions to global trajectory
        mu_global, mu_variance_global, _, _ = tp_kmp_model.fuse_predictions(
            pred_frames_obj=pred_frames_obj,
            sigma_pred=sigma_pred_local,
            mu_pred=mu_pred_local,
            sigma_pred_epistemic=sigma_pred_epistemic_local,
            sigma_pred_aleatoric=sigma_pred_aleatoric_local,
        )

        ### Plotting ###
        tp_kmp_model.plot_demonstrations(
            mu_local=mu_pred_local,
            sigma_pred_local=sigma_pred_local,
            mu_global=mu_global,
            variance_global=mu_variance_global,
            show_frame_plots=True,
            output_file_path=Path(__file__).parent / "adding_via_points.svg",
            scale_std_factor=10,
            legend_row_index=2,
            show_plot=plot,
        )

        return pred_frames_obj, tp_kmp_model


if __name__ == "__main__":
    from interactive_incremental_learning import ConfigParams, initialize_tpkmp

    ################################################### Initialize Model
    config = ConfigParams()
    tp_kmp_model, tp_kmp_params, frames = initialize_tpkmp(config)

    ################################################### RUN Generalization EXPERIMENT
    pred_frames_obj, tp_kmp_model = AddViaPointsExperiment().run(
        tp_kmp_model=tp_kmp_model, tp_kmp_params=tp_kmp_params, frames_used_for_demonstrations=frames, params=config, plot=True
    )
