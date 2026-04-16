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
Script to run the experiment to add frames

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
from typing import ClassVar

import numpy as np

from interactive_incremental_learning import TPKMP, ConfigParams, TPKMPConfigParams
from interactive_incremental_learning.common.data_object import (
    DataObject,
    DataObjectType,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the frame-addition experiment (camera frame position)."""

    end_time_stamp: float = 1.0

    frames_data = np.array([[0.3294, 0.5512, 0.1555, 1.0, 0.0, 0.0, 0.0], [-0.107, -0.6954, 0.0672, 0.707, 0.0, 0.0, -0.707]])
    # camera close to robot base to scan ring before measuring station, Original used for videos
    frame_position_in_between: ClassVar[list[float]] = [
        0.3989,
        0.0085,
        0.0849,
        0.026,
        -0.6901,
        -0.7224,
        0.035,
    ]
    camera_position_in_between: ClassVar[list[float]] = frame_position_in_between


class AddFramesExperiment:
    """Experiment demonstrating dynamic addition of a third reference frame."""

    def run(
        self,
        tp_kmp_model: TPKMP,
        tp_kmp_params: TPKMPConfigParams,
        frames_used_for_demonstrations: list[DataObject],
        params: ConfigParams,
        plot: bool = False,
    ) -> tuple[DataObject, TPKMP]:
        """Adds a third reference frame (camera) and shows how the global trajectory adapts.

        :param tp_kmp_model: trained TP-KMP model
        :param tp_kmp_params: model hyperparameters
        :param frames_used_for_demonstrations: frames from the training demonstrations
        :param params: data structure configuration
        :param plot: if True, displays and saves the figure
        :return: tuple of (predicted DataObject, updated TPKMP model)
        """
        ### Add new object to the scene ###
        experiment_config = ExperimentConfig()

        ######## TP-KMP Prediction for time horizon ########
        x_test = np.linspace(0, experiment_config.end_time_stamp, tp_kmp_params.number_of_test_points)

        # First set placeholder Frame to position of new object #
        new_frame_obj_in_between = DataObject(
            data=np.array([experiment_config.camera_position_in_between]),
            data_structure=params.frames_data_structure,
            data_type=DataObjectType.ARRAY_2D,
        )

        times_for_in_between = [0.5, 0.495, 0.505]

        # Set new object positions for trajectory prediction
        if experiment_config.frames_data is None:
            pred_frames_data = frames_used_for_demonstrations[0]
        else:
            pred_frames_data = experiment_config.frames_data
        pred_frames_obj = DataObject(
            data=pred_frames_data, data_structure=params.frames_data_structure, data_type=DataObjectType.ARRAY_2D
        )
        # Add frame in between
        tp_kmp_model.add_placeholder_kmp(x_test=x_test)
        # Add via point to new frame
        tp_kmp_model.add_via_point_to_placeholder_frame(
            times=times_for_in_between,
            viapoint_position=experiment_config.frame_position_in_between,
            new_frame_obj=new_frame_obj_in_between,
            frame_index=2,
        )
        pred_frames_obj = pred_frames_obj.concatenate_multiple([new_frame_obj_in_between])  # add new Frame

        # Calculate prediction
        mu_pred_local, sigma_pred_local, sigma_pred_epistemic_local, sigma_pred_aleatoric_local = tp_kmp_model.predict(
            x_test=x_test
        )

        # Combine local predictions to global trajectory
        mu_global, mu_variance_global, _, _ = tp_kmp_model.fuse_predictions(
            pred_frames_obj=pred_frames_obj,
            sigma_pred=sigma_pred_local,
            mu_pred=mu_pred_local,
            sigma_pred_aleatoric=sigma_pred_aleatoric_local,
            sigma_pred_epistemic=sigma_pred_epistemic_local,
        )

        ### Plotting ###
        tp_kmp_model.plot_demonstrations(
            mu_local=mu_pred_local,
            sigma_pred_local=sigma_pred_local,
            mu_global=mu_global,
            variance_global=mu_variance_global,
            show_frame_plots=True,
            output_file_path=Path(__file__).parent / "adding_frames.svg",
            scale_std_factor=10,
            legend_row_index=2,
            show_plot=plot,
            exclude_axis_from_span=[2],
        )

        return pred_frames_obj, tp_kmp_model


if __name__ == "__main__":
    from interactive_incremental_learning import initialize_tpkmp

    ################################################### Initialize Model
    config = ConfigParams()
    tp_kmp_model, tp_kmp_params, frames = initialize_tpkmp(config)

    ################################################### RUN Generalization EXPERIMENT
    pred_frames_obj, tp_kmp_model = AddFramesExperiment().run(
        tp_kmp_model=tp_kmp_model, tp_kmp_params=tp_kmp_params, frames_used_for_demonstrations=frames, params=config, plot=True
    )
