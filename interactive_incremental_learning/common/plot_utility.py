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
Script for plottings

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

from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    # Prevent circular imports
    from interactive_incremental_learning import DataObject

from interactive_incremental_learning.common.data_object import InputType


class PlotUtility:
    """Static utility class for plotting GMM/GMR results with 2D and 3D visualizations."""

    colors: ClassVar[list[str]] = ["navy", "turquoise", "darkorange"]
    inited = False
    figure = None
    plot_gmm_and_gmr = True
    amount_of_demos = -1

    @staticmethod
    def make_single_ellipse(means: list, covariances: np.ndarray, color: str, ax: Any):
        """
        Creates a visualisation of a single GMM by a ellipse and adds it into a given 2d subplot.
        :param means: Mean of all components of the GMM
        :param covariances: Covariances of all components of the GMM
        :param color: Predefined color for the ellipse.
        :param ax: mpl.axes._subplots.AxesSubplot
        """
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse((means[0], means[1]), v[0], v[1], angle=180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

    @staticmethod
    def make_single_ellipsoid(means: list, covariances: np.ndarray, color: str, ax: Any, draw_wireframe: bool = True):
        """
        Creates a visualisation of a single GMM by a ellipsoid and adds it into a given 3d subplot.
        :param means: Mean of all components fo the GMM.
        :param covariances: Covariances of all components of the GMM.
        :param color: Predefined color for the ellipsoid
        :param ax: mpl.axes._subplots.AxesSubplot
        """
        # v = length of vector
        # w = eigenvector
        v, w = np.linalg.eigh(covariances)
        # convert to variance
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)

        u = []
        for i in range(3):
            u.append(w[i] / np.linalg.norm(w[i]))

        # Cartesian coordinates that correspond to the spherical angles:
        # (this is the equation of an ellipsoid):
        resolution = 20
        q = np.linspace(0, 2 * np.pi, resolution)
        r = np.linspace(0, np.pi, resolution)
        x = v[0] * np.outer(np.cos(q), np.sin(r))
        y = v[1] * np.outer(np.sin(q), np.sin(r))
        z = v[2] * np.outer(np.ones_like(q), np.cos(r))

        final_poses_list = []
        rot_mat = np.array([u[0], u[1], u[2]])
        # flatten the values to rotate them
        x = x.reshape([-1])
        y = y.reshape([-1])
        z = z.reshape([-1])
        for xyz in zip(x, y, z):
            final_poses_list.append(np.dot(rot_mat, np.array(xyz)))
        # add the mean after rotation
        final_poses = np.array(final_poses_list)
        final_poses += means
        # get the x, y, z value from the final pose
        x = final_poses[:, 0]
        y = final_poses[:, 1]
        z = final_poses[:, 2]
        # reshape back to the resolution size
        x = x.reshape((resolution, resolution))
        y = y.reshape((resolution, resolution))
        z = z.reshape((resolution, resolution))

        # Plot:
        ax.plot_surface(x, y, z, color=color, alpha=0.2)
        if draw_wireframe:
            ax.plot_wireframe(x, y, z, color=color, linewidth=0.1)

    @staticmethod
    def plot_gmm_variances(
        means: "DataObject | None",
        covariances: "DataObject | None",
        n_components: int,
        plot_axis: int,
        used_colors: list,
        used_axis: Any,
        data_to_plot: list[InputType],
        plot_dimensions="2d",
    ):
        """
        Plots the variances for all components of the GMM.
        :param means: Mean of all components of the GMM.
        :param covariances: Covariances of all components of the GMM.
        :param n_components: Amount of GMM components (Gausians).
        :param plot_axis: Int value indicatng the plot in the figure.
        :param used_colors: Prefedined colors to use.
        :param used_axis: mpl.axes._subplots.AxesSubplot
        :param data_to_plot: Indicating what data you want to plot, e. g. [x, y, z]
        """
        if means is None or covariances is None:
            return
        for i in range(n_components):
            if plot_dimensions == "2d":
                cc = covariances.get_array([InputType.T, data_to_plot[plot_axis]])[i]
                calc_means = means.get_array([InputType.T, data_to_plot[plot_axis]])[i]
                current_color = used_colors[i]
                PlotUtility.make_single_ellipse(means=calc_means, covariances=cc, color=current_color, ax=used_axis)
            elif plot_dimensions == "3d":
                current_color = used_colors[i]
                cc = covariances.get_array(data_to_plot)[i]
                calc_means = means.get_array(data_to_plot)[i]
                PlotUtility.make_single_ellipsoid(means=calc_means, covariances=cc, color=current_color, ax=used_axis)

    @staticmethod
    def draw_TPGMM(
        n_components: int,
        test_data: np.ndarray,
        predicted_points: "DataObject",
        std_predicted_points: np.ndarray,
        train_data: "DataObject",
        covariances: "DataObject | None",
        means: "DataObject | None",
        data_to_plot: list[InputType],
        plot_dimensions: str = "2d",
        amount_of_plots: int = 2,
        title: str | None = None,
        additional_trajectory: "DataObject | None" = None,
        frame_position: "DataObject | None" = None,
        global_via_points: "list[DataObject] | None" = None,
    ):
        """
        Draws the regression function GMR including the area of the variance.
        :param n_components: Number of Gaussians in the GMM.
        :param test_data: DataObject with the target values (Input used for GMR)
        :param predicted_points: Predicted values from GMR
        :param std_predicted_points: std deviation from the predicted values of GMR
        :param gmm: The Gausian Mixture Model class
        :param data_to_plot: Indicating what data you want to plot, e. g. [x, y, z]
        :param plot_dimensions: String defining if it is a 2d or 3d plot. Default is 2d.
        :param amount_of_plots: Int defining amount of plots in the given figure. Default is 2.
        :param title: Title of the figure.
        :param additional_trajectory: An additional trajectory one want to add to the plots (e.g. local vs global predictions).
        :param global_via_points: If set the given via-points are added to the plot
        """
        if n_components <= 10:
            used_colors = list(mcolors.TABLEAU_COLORS.keys())
        else:
            used_colors = ["red"] * n_components

        if plot_dimensions == "3d":
            raise Exception("3d modus not implemented yet for TP GMM")

        else:
            PlotUtility._plot_axis(
                train_data=train_data,
                data_to_plot=data_to_plot,
                test_data=test_data,
                predicted_points=predicted_points,
                std_predicted_points=std_predicted_points,
                means=means,
                covariances=covariances,
                n_components=n_components,
                used_colors=used_colors,
                mode=["mean"],
                amount_of_plots=amount_of_plots,
                plot_dimensions=plot_dimensions,
                title=title,
                additional_trajectory=additional_trajectory,
                frame_point=frame_position,
                global_via_points=global_via_points,
            )
            PlotUtility._plot_axis(
                train_data=train_data,
                data_to_plot=data_to_plot,
                test_data=test_data,
                predicted_points=predicted_points,
                std_predicted_points=std_predicted_points,
                means=means,
                covariances=covariances,
                n_components=n_components,
                used_colors=used_colors,
                mode=["mean", "variance"],
                amount_of_plots=amount_of_plots,
                plot_dimensions=plot_dimensions,
                title=title,
                additional_trajectory=additional_trajectory,
                global_via_points=global_via_points,
            )

        plt.show()
        if PlotUtility.figure:
            PlotUtility.figure = None

    @staticmethod
    def _plot_axis(  # noqa: C901
        train_data: "DataObject",
        test_data: np.ndarray | None,
        predicted_points: "DataObject | None",
        std_predicted_points: np.ndarray | None,
        means: "DataObject | None",
        covariances: "DataObject | None",
        n_components: int,
        used_colors: list,
        mode: list[str],
        amount_of_plots: int,
        data_to_plot: list[InputType],
        recorded_traj: np.ndarray | None = None,
        plot_dimensions: str = "2d",
        title: str | None = None,
        additional_trajectory: "DataObject | None" = None,
        frame_point: "DataObject | None" = None,
        global_via_points: "list[DataObject] | None" = None,
    ):
        """

        :param train_object: Structured data from demonstration.
        :param test_data: Input data used for GMR, None if GMR was not used
        :param predicted_points: Predicted values by GMR, None if GMR was not used
        :param std_predicted_points: Variance of predicted points by GMR, None if GMR was not used
        :param means: Means of the GMMs
        :param covariances: Covariances of the GMMs
        :param n_components: Number of Gaussians used
        :param used_colors: List of colors you want to use in the graphic
        :param mode: So far "mean", "variance", "gmm" and "logging" are available
        :param amount_of_plots: Number of plots to create
        :param data_to_plot: Indicating what data you want to plot, e. g. [x, y, z]
        :param recorded_traj: Execution data of the robot, None if no logging was used
        :param plot_dimensions: Either 2d or 3d plots are available
        :param title: Sets a title to the figure
        :param additional_trajectory: An aditional trajectory one want to add to the plots (e. g. local vs global predictions).
        :param global_via_points: If set the given via-points are added to the plot
        """
        if plot_dimensions == "3d" and ("demonstration_data" in mode or "gmm" in mode):
            train_data_array = train_data.get_array(data_to_plot)
            if PlotUtility.figure is None:
                PlotUtility.figure = plt.figure(figsize=(24, 12))
            h = PlotUtility.figure.add_subplot(111, projection="3d")
            if "demonstration_data" in mode:
                h.scatter(
                    xs=train_data_array[:, 0],
                    ys=train_data_array[:, 1],
                    zs=train_data_array[:, 2],
                    s=1,
                )
                h.set_xlabel(data_to_plot[0])
                h.set_ylabel(data_to_plot[1])
                h.set_zlabel(data_to_plot[2])
            else:
                raise Exception("So far only demonstration data is supported for 3d")

            if "gmm" in mode:
                PlotUtility.plot_gmm_variances(
                    means=means,
                    covariances=covariances,
                    n_components=n_components,
                    plot_axis=0,
                    used_colors=used_colors,
                    used_axis=h,
                    plot_dimensions=plot_dimensions,
                    data_to_plot=data_to_plot,
                )
            if "mean" in mode and predicted_points is not None:
                predicted_points_3d = predicted_points.get_array(data_to_plot)
                h.scatter(
                    xs=predicted_points_3d[:, 0],
                    ys=predicted_points_3d[:, 1],
                    zs=predicted_points_3d[:, 2],
                )
                if additional_trajectory is not None:
                    additional_trajectory_array = additional_trajectory.get_array(data_to_plot)
                    h.scatter(
                        xs=additional_trajectory_array[:, 0],
                        ys=additional_trajectory_array[:, 1],
                        zs=additional_trajectory_array[:, 2],
                        color="red",
                    )
            if title:
                title = title + " 3d"
            else:
                title = "3d"
            h.set_title(title)
        else:
            train_data_array = train_data.get_array([InputType.T, *data_to_plot])
            assert predicted_points is not None
            predicted_points_array = predicted_points.get_array(data_to_plot)
            for plot_axis in range(1, len(data_to_plot) + 1):
                h = PlotUtility.get_subplot(
                    axes_nr=plot_axis,
                    mode=mode,
                    amount_of_variables=len(data_to_plot),
                    amount_of_plots=amount_of_plots,
                )
                if not title:
                    title = "2d"
                h.set_title(title)
                if not (("mean" in mode and "variance" in mode) or "logging" in mode):
                    if "demonstration_data" in mode:
                        if plot_dimensions == "3d":
                            h.scatter(
                                xs=train_data_array[:, 0],
                                ys=train_data_array[:, 1],
                                zs=train_data_array[:, 2],
                                s=1,
                            )
                        elif PlotUtility.amount_of_demos != -1:
                            split_data = np.reshape(
                                train_data_array,
                                (
                                    PlotUtility.amount_of_demos,
                                    -1,
                                    train_data_array.shape[-1],
                                ),
                            )
                            for index, current_split in enumerate(split_data):
                                h.scatter(
                                    x=current_split[:, 0],
                                    y=current_split[:, plot_axis],
                                    s=1,
                                    color=used_colors[index],
                                )
                        else:
                            h.scatter(
                                x=train_data_array[:, 0],
                                y=train_data_array[:, plot_axis],
                                s=1,
                            )
                    else:
                        if plot_dimensions == "2d":
                            h.scatter(
                                x=train_data_array[:, 0],
                                y=train_data_array[:, plot_axis],
                                s=1,
                            )
                        elif plot_dimensions == "3d":
                            h.scatter(
                                xs=train_data_array[:, 0],
                                ys=train_data_array[:, 1],
                                zs=train_data_array[:, 2],
                            )
                min_val, max_val = np.min(train_data_array[:, plot_axis]), np.max(train_data_array[:, plot_axis])
                if "mean" in mode:
                    h.scatter(
                        x=test_data,
                        y=predicted_points_array[:, plot_axis - 1],
                        c="#ff7f0e",  # orange
                        s=2,
                    )
                    _miv_val, max_val = (
                        np.min([np.min(predicted_points_array[:, plot_axis - 1]), min_val]),
                        np.max([np.max(predicted_points_array[:, plot_axis - 1]), max_val]),
                    )
                    if additional_trajectory is not None:
                        additional_trajectory_array = additional_trajectory.get_array(data_to_plot)
                        h.scatter(x=test_data, y=additional_trajectory_array[:, plot_axis - 1], c="#d11111", s=2)  # red
                        _miv_val, max_val = (
                            np.min([np.min(additional_trajectory_array[:, plot_axis - 1]), min_val]),
                            np.max([np.max(additional_trajectory_array[:, plot_axis - 1]), max_val]),
                        )
                    if frame_point is not None:
                        frame_point_array = frame_point.get_array(data_to_plot)
                        assert frame_point_array.shape[0] == 1, "Only one frame per drawing can be plotted."
                        h.axhline(frame_point_array[0, plot_axis - 1], c="g")
                        _miv_val, max_val = (
                            np.min([frame_point_array[0, plot_axis - 1], min_val]),
                            np.max([np.max(frame_point_array[0, plot_axis - 1]), max_val]),
                        )

                    if std_predicted_points is not None and "variance" in mode:
                        upper_line = predicted_points_array[:, plot_axis - 1] + std_predicted_points[:, plot_axis - 1]
                        under_line = predicted_points_array[:, plot_axis - 1] - std_predicted_points[:, plot_axis - 1]
                        assert test_data is not None
                        if np.array(test_data).ndim == 2:
                            if (np.shape(test_data)[1]) == 1:
                                _test_data = test_data[:, 0]
                        else:
                            _test_data = test_data
                        h.fill_between(
                            x=_test_data,
                            y1=under_line,
                            y2=upper_line,
                            alpha=0.3,
                            color="#080808",
                        )
                if "logging" in mode:
                    h.scatter(
                        x=test_data,
                        y=predicted_points_array[:, plot_axis - 1],
                        c="#ff7f0e",
                        s=2,
                    )
                    if std_predicted_points is not None:
                        upper_line = predicted_points_array[:, plot_axis - 1] + std_predicted_points[:, plot_axis - 1]
                        under_line = predicted_points_array[:, plot_axis - 1] - std_predicted_points[:, plot_axis - 1]
                        h.fill_between(
                            x=test_data,
                            y1=under_line,
                            y2=upper_line,
                            alpha=0.3,
                            color="#080808",
                        )
                    _miv_val, max_val = (
                        np.min([np.min(predicted_points_array[:, plot_axis - 1]), min_val]),
                        np.max([np.max(predicted_points_array[:, plot_axis - 1]), max_val]),
                    )

                    # bring recorded traj to same lengh as test_data
                    assert recorded_traj is not None
                    assert test_data is not None
                    len_recorded_traj = len(recorded_traj)
                    time_step_size_recorded = (np.max(test_data) - np.min(test_data)) / float(len_recorded_traj)
                    recorded_traj_time = [time_step_size_recorded * i for i in range(len_recorded_traj)]
                    h.scatter(x=recorded_traj_time, y=recorded_traj[:, plot_axis], s=2)
                    _miv_val, max_val = (
                        np.min([np.min(recorded_traj[:, plot_axis - 1]), min_val]),
                        np.max([np.max(recorded_traj[:, plot_axis - 1]), max_val]),
                    )

                h.set_ylabel(str(data_to_plot[plot_axis - 1]))
                if "gmm" in mode:
                    PlotUtility.plot_gmm_variances(
                        means=means,
                        covariances=covariances,
                        n_components=n_components,
                        plot_axis=plot_axis - 1,
                        used_colors=used_colors,
                        used_axis=h,
                        data_to_plot=data_to_plot,
                        plot_dimensions=plot_dimensions,
                    )

                if global_via_points is not None:
                    # Show via-points as x-es
                    for via_point in global_via_points:
                        via_points_array = via_point.get_array([InputType.T, *data_to_plot])
                        h.plot(via_points_array[0, 0], via_points_array[0, plot_axis], "gx")  # , markersize=25)

                if max_val - min_val > 0:
                    max_val - min_val
                    # h.set_ylim([min_val - diff * 0.1, max_val + diff * 0.1])

    @staticmethod
    def init():
        """Initialize the shared matplotlib figure if not already created."""
        if not PlotUtility.inited or PlotUtility.figure is None:
            PlotUtility.figure = plt.figure(figsize=(24, 10))
            PlotUtility.inited = True

    @staticmethod
    def get_subplot(
        axes_nr: int,
        mode: list[str],
        amount_of_variables: int,
        amount_of_plots: int,
    ) -> plt.Axes:
        """
        Get the subplot to plot the new graphics in.
        :param axes_nr: index of the current plot axes
        :param mode: What is shown in the plot ["demonstration_data","gmm","mean","logging"]
        :param amount_of_variables: E. g. 3 if the variables are ["x","y","z"] -> number of rows
        :param amount_of_plots: number of columns e. g. 5 if you want to show
            "demonstration_data", "gmm", "mean", "variance" and "logging"
        :return: Corresponding subplot
        """
        PlotUtility.init()
        if PlotUtility.inited:
            current_plot_nr = -1
            if "demonstration_data" in mode:
                current_plot_nr = 0
            elif "gmm" in mode:
                current_plot_nr = min(1, amount_of_plots - 1)
            elif "mean" in mode and "variance" not in mode:
                if amount_of_plots > 2:
                    current_plot_nr = min(2, amount_of_plots - 1)
                else:
                    current_plot_nr = 0
            elif "mean" in mode and "variance" in mode:
                if amount_of_plots > 3:
                    current_plot_nr = min(3, amount_of_plots - 1)
                elif amount_of_plots == 3:
                    current_plot_nr = min(2, amount_of_plots - 1)
                elif amount_of_plots == 2:
                    current_plot_nr = 1
                else:
                    current_plot_nr = 0
            elif "logging" in mode:
                current_plot_nr = amount_of_plots - 1
            else:
                raise ValueError(f"The mode is unknown: {mode}")
            new_nr = (axes_nr - 1) * amount_of_plots + current_plot_nr + 1
            assert PlotUtility.figure is not None
            h = PlotUtility.figure.add_subplot(amount_of_variables, amount_of_plots, new_nr)
            if axes_nr == 1:
                h.set_title(" ".join(e.replace("_", " ") for e in mode))
            return h
        else:
            raise RuntimeError("The PlotData has to be inited before")
