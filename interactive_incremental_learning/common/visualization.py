"""Visualization utilities for TP-KMP experiment results.

Provides plotting functions for local/global trajectory predictions with uncertainty
bands, via-point annotations, and variable stiffness profiles.
"""

from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


class TpkmpDataClass:
    """Container for TP-KMP prediction data used by the plotting functions."""

    def __init__(
        self,
        local_mu: np.ndarray,
        global_mu: np.ndarray,
        local_std: np.ndarray,
        global_std: np.ndarray,
        demonstration: np.ndarray,
        local_demonstration: np.ndarray,
        local_std_epistemic: np.ndarray,
        global_std_epistemic: np.ndarray,
        frames: np.ndarray,
        input_s: np.ndarray,
        local_viapoints_f1: np.ndarray,
        local_viapoints_f2: np.ndarray,
        local_viapoints_f3: np.ndarray,
        global_viapoints: np.ndarray,
        global_variance: np.ndarray | None = None,
        global_variance_epistemic: np.ndarray | None = None,
        Gp: np.ndarray | None = None,
    ):

        self.local_mu = np.array(local_mu)
        self.global_mu = np.array(global_mu)
        self.demonstrations = np.array(demonstration)
        self.local_demonstrations = np.array(local_demonstration)
        self.local_std = np.array(local_std)
        self.local_std_epistemic = np.array(local_std_epistemic)
        self.global_std = np.array(global_std)
        self.global_std_epistemic = np.array(global_std_epistemic)
        self.global_variance = np.array(global_variance)
        self.global_variance_epistemic = np.array(global_variance_epistemic)
        self.frames = np.array(frames)
        self.input_s = np.array(input_s)
        self.local_viapoints_f1 = np.array(local_viapoints_f1)  # if non, list of one 0 given
        self.local_viapoints_f2 = np.array(local_viapoints_f2)  # if non, list of one 0 given
        self.local_viapoints_f3 = np.array(local_viapoints_f3)  # if non, list of one 0 given
        self.global_viapoints = np.array(global_viapoints)  # if non, list of one 0 given
        self.Gp = Gp


def plot_demonstrations(  # noqa: C901
    tpkmp_data: TpkmpDataClass,
    output_file_path: Path,
    show_frame_plots: bool = False,
    show_frame_3: bool = False,
    show_via_points: bool = False,
    exclude_axis_from_span: list[int] | None = None,
    legend_loc: int | None = None,
    legend_row_index: int = 0,
    legend_font_size: float = 9,
    scale_std_factor=1,
    figure_hight: int = 4,
    figure_width: int = 10,
):
    "default for all figures, shows all recorded demonstrations globally including mean, covar"

    sns.set_theme()

    demonstrations = tpkmp_data.demonstrations
    if show_frame_plots and show_frame_3:
        raise ValueError("Only one of them can be true at once")

    titles = ["Box 1 frame", "Box 2 frame", "Global frame"]

    if show_frame_plots:
        all_demonstrations = [
            (
                tpkmp_data.local_demonstrations[0],
                tpkmp_data.local_mu[0],
                tpkmp_data.local_std[0],
                None,
                tpkmp_data.local_viapoints_f1,
                None,
            ),
            (
                tpkmp_data.local_demonstrations[1],
                tpkmp_data.local_mu[1],
                tpkmp_data.local_std[1],
                None,
                tpkmp_data.local_viapoints_f2,
                None,
            ),
            (demonstrations, tpkmp_data.global_mu, tpkmp_data.global_std, None, tpkmp_data.global_viapoints, None),
        ]
    elif show_frame_3:
        all_demonstrations = [
            (None, tpkmp_data.local_mu[2], tpkmp_data.local_std[2], None, tpkmp_data.local_viapoints_f3, None),
            (demonstrations, tpkmp_data.global_mu, tpkmp_data.global_std, None, tpkmp_data.global_viapoints, None),
        ]
        titles = ["Camera frame", "Global frame"]

    else:
        all_demonstrations = [
            (demonstrations, tpkmp_data.global_mu, tpkmp_data.global_std, None, tpkmp_data.global_viapoints, None)
        ]

    # change this numbers to change font size
    plt.figure(figsize=(figure_width, figure_hight))
    red_color = "red"
    orange_color = "orange"
    green_color = "green"
    blue_color = "blue"
    black_color = "black"
    all_axis = []
    all_center_poses = []
    max_span = 0.0
    for j, (
        current_demonstrations,
        current_mu,
        current_std,
        current_std_epistemic,
        current_via_points,
        current_robot_movement,
    ) in enumerate(all_demonstrations):
        ax = []
        center_poses = []
        for i in range(3):
            min_value = np.inf
            max_value = -np.inf
            ax.append(plt.subplot(3, len(all_demonstrations), i * len(all_demonstrations) + 1 + j))
            plt.locator_params(axis="x", nbins=3)
            ax[-1].tick_params(axis="y", which="major", pad=-1)
            ax[-1].tick_params(axis="x", which="major", pad=-1)
            # plot demonstrations

            if current_demonstrations is not None:
                for x in range(current_demonstrations.shape[0]):
                    plt.plot(
                        current_demonstrations[x, :, 0],
                        current_demonstrations[x, :, i + 1],
                        color=black_color,
                        linestyle=":",
                        linewidth=1,
                        label="Demonstrations" if x == 0 else None,
                    )
                min_value = np.min([min_value, np.min(current_demonstrations[:, :, i + 1])])
                max_value = np.max([max_value, np.max(current_demonstrations[:, :, i + 1])])

            if current_robot_movement is None:
                # plot mu - we only show mu if we don't plot robot movement.
                plt.plot(tpkmp_data.input_s, current_mu[:, i], color=red_color, label="Prediction")

            # plot variance
            number_of_legend_cols = 2
            if current_std_epistemic is None:
                plt.fill_between(
                    tpkmp_data.input_s,
                    current_mu[:, i] - current_std[:, i] * scale_std_factor,
                    current_mu[:, i] + current_std[:, i] * scale_std_factor,
                    color=red_color,
                    alpha=0.2,
                    label="Std. Dev.",
                )
                current_std[:, i][np.isnan(current_std[:, i])] = 0.0
                min_value = np.min([min_value, np.min(current_mu[:, i] + current_std[:, i] * scale_std_factor)])
                min_value = np.min([min_value, np.min(current_mu[:, i] - current_std[:, i] * scale_std_factor)])
                max_value = np.max([max_value, np.max(current_mu[:, i] + current_std[:, i] * scale_std_factor)])
                max_value = np.max([max_value, np.max(current_mu[:, i] - current_std[:, i] * scale_std_factor)])
            else:
                current_std_epistemic[np.isnan(current_std_epistemic)] = 0.0
                aleatoric_std = current_std - np.abs(current_std_epistemic)

                plt.fill_between(
                    tpkmp_data.input_s,
                    current_mu[:, i] - aleatoric_std[:, i] * scale_std_factor,
                    current_mu[:, i] + aleatoric_std[:, i] * scale_std_factor,
                    color=green_color,
                    alpha=0.2,
                    label="Aleatoric",
                )
                plt.fill_between(
                    tpkmp_data.input_s,
                    current_mu[:, i] - current_std_epistemic[:, i] * scale_std_factor,
                    current_mu[:, i] + current_std_epistemic[:, i] * scale_std_factor,
                    color=blue_color,
                    alpha=0.2,
                    label="Epistemic",
                )

                aleatoric_std[:, i][np.isnan(aleatoric_std[:, i])] = 0.0
                min_value = np.min([min_value, np.min(current_mu[:, i] + aleatoric_std[:, i] * scale_std_factor)])
                min_value = np.min([min_value, np.min(current_mu[:, i] - aleatoric_std[:, i] * scale_std_factor)])
                max_value = np.max([max_value, np.max(current_mu[:, i] + aleatoric_std[:, i] * scale_std_factor)])
                max_value = np.max([max_value, np.max(current_mu[:, i] - aleatoric_std[:, i] * scale_std_factor)])
                min_value = np.min([min_value, np.min(current_mu[:, i] + current_std_epistemic[:, i] * scale_std_factor)])
                min_value = np.min([min_value, np.min(current_mu[:, i] - current_std_epistemic[:, i] * scale_std_factor)])
                max_value = np.max([max_value, np.max(current_mu[:, i] + current_std_epistemic[:, i] * scale_std_factor)])
                max_value = np.max([max_value, np.max(current_mu[:, i] - current_std_epistemic[:, i] * scale_std_factor)])
                number_of_legend_cols = 2

            # Plot via-points
            if show_via_points:
                viapoints = np.reshape(current_via_points.copy(), (-1, 8))
                viapoints = viapoints[np.argsort(viapoints[:, 0])]
                min_dist = 0.01
                start_timestamp = None
                remove_indices: list[int] = []
                for index, timestamp in enumerate(viapoints[:, 0]):
                    if start_timestamp is None:
                        start_timestamp = timestamp
                    elif timestamp - start_timestamp < min_dist:
                        remove_indices.append(index)
                    else:
                        start_timestamp = timestamp
                selection = np.ones(viapoints.shape[0], dtype=bool)
                selection[remove_indices] = False
                viapoints = viapoints[selection]
                plt.scatter(
                    viapoints[:, 0],
                    viapoints[:, i + 1],
                    edgecolors=orange_color,
                    marker="o",
                    label="Via points",
                    facecolors="none",
                )

            # plot robot movement
            if current_robot_movement is not None:
                plt.plot(tpkmp_data.input_s, current_robot_movement[:, i], color=red_color, label="Robot movement")

            if j == 0:
                curr_label = ["x [m]", "y [m]", "z [m]"]
                plt.ylabel(curr_label[i])
            if i == 0:
                plt.title(titles[j])
            if (exclude_axis_from_span is not None and j not in exclude_axis_from_span) or exclude_axis_from_span is None:
                max_span = np.max([max_span, max_value - min_value])
            center_poses.append((max_value + min_value) * 0.5)
            if i == legend_row_index and j == len(all_demonstrations) - 1:
                plt.legend(loc=legend_loc, prop={"size": legend_font_size}, ncol=number_of_legend_cols)

        if i > 1:
            ax[i].sharex(ax[0])
        for i in range(2):
            ax[i].set_xticklabels([])
        all_axis.append(ax)
        all_center_poses.append(center_poses)
        plt.xlabel("time")
    half_span = max_span * 0.55
    for current_center_poses, current_axes in zip(all_center_poses, all_axis):
        for current_center_pose, current_axis in zip(current_center_poses, current_axes):
            current_center_pose = np.round(current_center_pose / 0.25) * 0.25
            # comment this line to remove same span on all figures
            current_axis.set_ylim((current_center_pose - half_span, current_center_pose + half_span))

    if show_frame_3:
        all_axis[0][0].set_ylim((-0.05, 0.05))
        all_axis[0][1].set_ylim((-0.05, 0.05))
        all_axis[0][2].set_ylim((-0.05, 0.05))
    plt.tight_layout(pad=0, h_pad=0.35)
    plt.savefig(str(output_file_path.absolute()), bbox_inches="tight")
    plt.show()


def plot_stiffness(
    tpkmp_data_01: TpkmpDataClass, tpkmp_data_02: TpkmpDataClass, tpkmp_data_03: TpkmpDataClass, output_file_path: Path
):
    """Plot variable stiffness profiles computed from epistemic/aleatoric uncertainty for three kernel lengths."""
    plt.figure(figsize=(5, 2))
    sns.set_theme()

    # calculate stiffness G_p
    plt.locator_params(axis="x", nbins=3)
    global_var = []
    global_var_epistemic = []
    global_var_aleatoric = []
    for tpkmp_data in [tpkmp_data_01, tpkmp_data_02, tpkmp_data_03]:
        curr_global_var = tpkmp_data.global_variance[:, :3]
        curr_global_var_epistemic = tpkmp_data.global_variance_epistemic[:, :3]
        curr_global_var_aleatoric = curr_global_var - curr_global_var_epistemic
        global_var.append(curr_global_var)
        global_var_epistemic.append(curr_global_var_epistemic)
        global_var_aleatoric.append(curr_global_var_aleatoric)

    reg = 1.5e-3
    c1 = 5000
    c2 = 0.00015

    w1 = []
    w2 = []
    for i in range(len(global_var_epistemic)):
        curr_w1 = 1 / (1 + np.exp(-c1 * (global_var_epistemic[i] - c2)))
        w2.append(1 - curr_w1)
        w1.append(curr_w1)

    gamma_ep = 1000.0
    gamma_al = 1

    Gp = []
    for i in range(len(global_var_epistemic)):
        Gp.append(
            w1[i] * (gamma_ep * (global_var_epistemic[i] + reg)) ** (-1)
            + w2[i] * (gamma_al * (global_var_aleatoric[i] + reg)) ** (-1)
        )

    Gp_arr = np.array(Gp)
    plt.title("Stiffness (Gp) using uncertainty split")

    plt.ylabel("Gp[x] in [N/m]")

    # only plot one axis
    labels = ["l = 0.1", "l = 0.2", "l = 0.3"]
    for i in range(3):
        plt.plot(tpkmp_data_01.input_s, np.min(Gp_arr[i, 0, :, :3], axis=1), label=labels[i])

    plt.legend(prop={"size": 9})
    plt.tight_layout()
    plt.savefig(str(output_file_path.absolute()), bbox_inches="tight")
    plt.show()
