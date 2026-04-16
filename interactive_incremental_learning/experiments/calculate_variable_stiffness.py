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
Script to run the experiment which calculates the variable stiffness

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

import matplotlib.pyplot as plt
import numpy as np

from interactive_incremental_learning import ConfigParams, initialize_tpkmp
from interactive_incremental_learning.common.data_object import (
    DataObject,
    DataObjectType,
)
from interactive_incremental_learning.common.tp_kmp import TPKMP, TPKMPConfigParams


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the variable stiffness experiment (extended time horizon)."""

    end_time_stamp: float = 1.25  # Extend beyond training range (0-1) to show out-of-distribution behavior
    frames_data = None


def _compute_stiffness(tp_kmp_model, x_test, pred_frames_obj, params):
    """Compute original and separated stiffness profiles for a trained model.

    Uses the paper's eq. (4) decomposition: Sigma_ep = k** - k* K^{-1} k*^T (pure kernel inverse)
    and eq. (5) stiffness formula: Gp = w1*(delta_ep*Sigma_ep*)^{-1} + w2*(delta_al*Sigma_al*)^{-1}
    with sigma_ep^2 = tr(Sigma_ep*)/D_out as scalar sigmoid input.
    """
    # Paper eq. (4) uses K^{-1} for epistemic (no regularization).
    # Set epi_reg to near-zero for numerical stability.
    for kmp in tp_kmp_model.tp_kmp:
        n = (kmp.N + kmp.nb_via) * kmp.nb_dim_out
        kmp.epi_reg = 1e-8
        kmp.invK_epi = np.linalg.inv(kmp.K + 1e-8 * np.eye(n))

    mu_pred_local, sigma_pred_local, sigma_pred_epistemic_local, sigma_pred_aleatoric_local = tp_kmp_model.predict(
        x_test=x_test
    )

    mu_global, mu_variance_global, mu_variance_epistemic_global, mu_variance_aleatoric_global = tp_kmp_model.fuse_predictions(
        pred_frames_obj=pred_frames_obj,
        sigma_pred=sigma_pred_local,
        mu_pred=mu_pred_local,
        sigma_pred_aleatoric=sigma_pred_aleatoric_local,
        sigma_pred_epistemic=sigma_pred_epistemic_local,
    )

    global_variance_epistemic = tp_kmp_model.get_variance_of_covariance(
        mu_TP=mu_global, mu_variance=mu_variance_epistemic_global
    )
    global_variance_total = tp_kmp_model.get_variance_of_covariance(mu_TP=mu_global, mu_variance=mu_variance_global)
    global_variance_aleatoric = tp_kmp_model.get_variance_of_covariance(
        mu_TP=mu_global, mu_variance=mu_variance_aleatoric_global
    )

    # Paper hyperparameters (Table I in appendix)
    reg = 1.5e-3
    c1 = 5000
    c2 = 1.5e-3  # Paper: 1.5e-3
    gamma_ep = 1000.0
    gamma_al = 1
    pos_dim = 3

    # Paper: sigma_ep^2 = tr(Sigma_ep*)/D_out -- scalar per time step
    sigma_ep_sq = np.mean(global_variance_epistemic[:, :pos_dim], axis=1)
    w1 = 1 / (1 + np.exp(-c1 * (sigma_ep_sq - c2)))
    w2 = 1 - w1

    # Stiffness for x-dimension (paper eq. 5, applied element-wise for diagonal covariance)
    Gp_x = w1 * (gamma_ep * (global_variance_epistemic[:, 0] + reg)) ** (-1) + w2 * (
        gamma_al * (global_variance_aleatoric[:, 0] + reg)
    ) ** (-1)

    original_stiffness = 1.0 / (global_variance_total[:, 0] + reg)
    separated_stiffness = Gp_x

    return x_test, original_stiffness, separated_stiffness, mu_global, None


class CalculateVariableStiffnessExperiment:
    """Experiment computing adaptive stiffness profiles from epistemic/aleatoric uncertainty decomposition."""

    def run(
        self,
        tp_kmp_model: TPKMP,
        tp_kmp_params: TPKMPConfigParams,
        frames_used_for_demonstrations: list[DataObject],
        params: ConfigParams,
        plot: bool = False,
    ):
        """Computes variable stiffness profiles from epistemic/aleatoric uncertainty (Figures 7 & 8).

        Trains models with different kernel lengths, computes stiffness before and after
        adding a via-point at t=1.10, and plots the comparison.

        :param tp_kmp_model: trained TP-KMP model (used only for frame configuration)
        :param tp_kmp_params: model hyperparameters (number_of_test_points is used)
        :param frames_used_for_demonstrations: frames from the training demonstrations
        :param params: data structure configuration
        :param plot: if True, displays and saves the figures
        """
        experiment_config = ExperimentConfig()
        x_test = np.linspace(0, experiment_config.end_time_stamp, tp_kmp_params.number_of_test_points)

        # Set prediction frames
        if experiment_config.frames_data is None:
            frame_data_obj = frames_used_for_demonstrations[0]
            pred_frames_data = frame_data_obj.get_array(frame_data_obj.get_data_structure())
        else:
            pred_frames_data = experiment_config.frames_data
        pred_frames_obj = DataObject(
            data=pred_frames_data, data_structure=params.frames_data_structure, data_type=DataObjectType.ARRAY_2D
        )

        # Compute stiffness for different kernel lengths
        kernel_lengths = [0.1, 0.2, 0.3]
        colors = ["blue", "orange", "green"]

        results_before = {}
        for l_val in kernel_lengths:
            kmp_params = TPKMPConfigParams(length_scale_of_the_kernel=l_val)
            model, _, _ = initialize_tpkmp(params, kmp_params)
            time_points, orig, sep, _mu_global, _Gp = _compute_stiffness(model, x_test, pred_frames_obj, params)
            results_before[l_val] = (time_points, orig, sep)

        # Get trajectory for robot control (from default kernel length model)
        time_points, _original_stiffness_before, _separated_stiffness_before = results_before[kernel_lengths[0]]

        ### Add via-point after t=1.0 and compute Figure 8 ###
        via_point_times = [1.10]
        via_point_positions = [
            np.array([0.5, 0.2, 0.25, 1.0, 0.0, 0.0, 0.0]),
        ]

        results_after = {}
        for l_val in kernel_lengths:
            kmp_params = TPKMPConfigParams(length_scale_of_the_kernel=l_val)
            model, _, _ = initialize_tpkmp(params, kmp_params)

            # Predict once to initialize KMP inputs before adding via-points
            model.predict(x_test=x_test)

            # Add via-points
            for time_stamp, position in zip(via_point_times, via_point_positions):
                via_point_obj = DataObject(
                    data=position.reshape(1, -1),
                    data_structure=params.robot_data_structure[1:],
                    data_type=DataObjectType.ARRAY_2D,
                )
                model.add_via_points_locally_to_kmp(
                    kmp_idx=0, time_stamps_for_insertions=[time_stamp], via_local=[via_point_obj], variance=1e-6
                )
                model.add_via_points_locally_to_kmp(
                    kmp_idx=1, time_stamps_for_insertions=[time_stamp], via_local=[via_point_obj], variance=1e-6
                )

            time_points, orig, sep, _, _ = _compute_stiffness(model, x_test, pred_frames_obj, params)
            results_after[l_val] = (time_points, orig, sep)

        ### Plotting ###
        if plot:
            import seaborn as sns

            sns.set_theme()

            # Figure 7: Before via-points
            _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.set_title("Stiffness ($G_P$) using combined uncertainty")
            ax2.set_title("Stiffness ($G_P$) using uncertainty split")

            for l_val, color in zip(kernel_lengths, colors):
                t, orig, sep = results_before[l_val]
                ax1.plot(t, orig, color=color, linewidth=2, label=f"l = {l_val}")
                ax2.plot(t, sep, color=color, linewidth=2, label=f"l = {l_val}")

            for ax in (ax1, ax2):
                ax.set_xlabel("Normalized trajectory progression s")
                ax.set_ylabel("Gp[x] in [N/m]")
                ax.grid(True, alpha=0.3)
                ax.legend()

            plt.tight_layout()
            plt.savefig(Path(__file__).parent / "stiffness_before_viapoints_figure7.svg", bbox_inches="tight")
            plt.show()

            # Figure 8: After via-points
            _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.set_title("Stiffness ($G_P$) using combined uncertainty\n(after via-point)")
            ax2.set_title("Stiffness ($G_P$) using uncertainty split\n(after via-point)")

            for l_val, color in zip(kernel_lengths, colors):
                t, orig, sep = results_after[l_val]
                ax1.plot(t, orig, color=color, linewidth=2, label=f"l = {l_val}")
                ax2.plot(t, sep, color=color, linewidth=2, label=f"l = {l_val}")

            for ax in (ax1, ax2):
                for vp_time in via_point_times:
                    ax.axvline(x=vp_time, color="red", linestyle="--", alpha=0.7, linewidth=1)
                ax.set_xlabel("Normalized trajectory progression s")
                ax.set_ylabel("Gp[x] in [N/m]")
                ax.grid(True, alpha=0.3)
                ax.legend()

            plt.tight_layout()
            plt.savefig(Path(__file__).parent / "stiffness_after_viapoints_figure8.svg", bbox_inches="tight")
            plt.show()


if __name__ == "__main__":
    from interactive_incremental_learning import initialize_tpkmp

    ################################################### Initialize Model
    config = ConfigParams()
    tp_kmp_model, tp_kmp_params, frames = initialize_tpkmp(config)

    ################################################### RUN Generalization EXPERIMENT
    CalculateVariableStiffnessExperiment().run(
        tp_kmp_model=tp_kmp_model,
        tp_kmp_params=tp_kmp_params,
        frames_used_for_demonstrations=frames,
        params=config,
        plot=True,
    )
