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
Script for Task-Parameterized Kernelized Movement Primitives

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

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

from interactive_incremental_learning.common.data_handler import DataHandler
from interactive_incremental_learning.common.data_object import (
    ORIENTATION_QUAT_W_FIRST,
    ORIENTATION_QUAT_W_LAST,
    ORIENTATION_SO3_MANIFOLD,
    ORIENTATION_VECTOR_6D,
    POSITION,
    DataObject,
    DataObjectType,
    InputType,
    RotationOutputType,
    assert_list_of_input_types,
)
from interactive_incremental_learning.common.kmp import Kmp
from interactive_incremental_learning.common.tp_math import make_skew_matrix_for_w_last

"""
Description
"""


@dataclass
class TPKMPConfigParams:
    k_gmm: int = 12

    length_scale_of_the_kernel: float = 0.1  # 0.3 # l
    proportional_kernel_scaling_factor: float = 1.0  # h
    lambda1: float = 0.1
    lambda2: float = 1  # 0.1 #How much trying to track the covariance distribution in the KMP
    alpha: float = 1  # 50.0
    kernel_function: str = "matern2"  # "rbf"

    frequency = 100.0  # Points per second

    number_of_test_points: int = 500  # N


class TPKMP:
    """Task-Parameterized Kernelized Movement Primitives.

    Learns trajectory distributions in frame-local coordinate systems using KMP
    and fuses them into a global prediction via precision-weighted Gaussian product.
    Supports incremental learning through via-points and dynamic frame addition.
    """

    def __init__(
        self,
        params: TPKMPConfigParams,
        data_handler: list[DataHandler],
        input_data_structure: list[InputType],
        output_data_structure: list[InputType],
        draw: bool = False,
    ):
        """
        :param params: hyperparameters for KMP and GMM
        :param data_handler: one DataHandler per demonstration, each containing trajectory data and reference frames
        :param input_data_structure: input dimensions for the model (e.g. time)
        :param output_data_structure: output dimensions for the model (e.g. position + orientation)
        :param draw: if True, enables visualization during training
        """
        self.params = params
        self.data_handler = data_handler
        self.input_data_structure = assert_list_of_input_types(input_data_structure)
        self.output_data_structure = assert_list_of_input_types(output_data_structure)
        self.nb_dim_in = len(self.input_data_structure)
        self.nb_dim_out = len(self.output_data_structure)
        if not all(
            dh.list_of_frames is not None and dh.list_of_frames.get_shape()[0] == self.number_of_frames for dh in data_handler
        ):
            raise ValueError("All data handlers must have the same number of frames!")
        self.draw = draw

        # The following variables are needed for training
        self.tp_kmp: list[Kmp] = []  # List to save all KMPs
        self.local_training_data_list: list[DataObject] = []
        self.local_training_data_non_merged_list: list[list[DataObject]] = []
        self.transformed_combined_datastructure: list[InputType] = []
        self.input_indices: list[int] = []  # For training
        self.output_indices: list[int] = []  # For training

        # For plotting
        self.local_via_points_per_frame: dict[int, list[DataObject]] = {}  # Dynamic frame storage
        self.global_via_points: list[DataObject] = []

    @property
    def number_of_frames(self) -> int:
        """Return the number of reference frames from the first data handler."""
        assert self.data_handler[0].list_of_frames is not None
        return self.data_handler[0].list_of_frames.get_shape()[0]

    def train(self):
        """Train one KMP per reference frame on the locally-transformed demonstration data."""

        for frame_index in range(self.data_handler[0].get_frames().get_shape()[0]):
            for_current_frame_data_objects: list[DataObject] = []
            for _index, multi_data_handler in enumerate(self.data_handler):
                new_data_obj = multi_data_handler.get_transformed_demonstration_for_frames_index(
                    index=frame_index,
                )
                new_data_obj.set_quaternions_positive()
                for_current_frame_data_objects.append(new_data_obj)
            self.local_training_data_non_merged_list.append(copy.deepcopy(for_current_frame_data_objects))
            local_data_object_for_current_frame: DataObject = for_current_frame_data_objects[0].concatenate_multiple(
                for_current_frame_data_objects[1:]
            )

            self.tp_kmp.append(
                Kmp(
                    gmm_n_components=self.params.k_gmm,
                    N=self.params.number_of_test_points,
                    length_scale=self.params.length_scale_of_the_kernel,
                    h=self.params.proportional_kernel_scaling_factor,
                    lambda1=self.params.lambda1,
                    lambda2=self.params.lambda2,
                    alpha=self.params.alpha,
                )
            )

            combined_datastructure = self.input_data_structure + self.output_data_structure
            datastructure_of_data = assert_list_of_input_types(local_data_object_for_current_frame.get_data_structure())
            all_input_types_present = all(input_type in datastructure_of_data for input_type in combined_datastructure)
            if all_input_types_present:
                self.transformed_combined_datastructure = combined_datastructure
                self.local_training_data_list.append(local_data_object_for_current_frame)
                local_data_as_np_array: np.ndarray = local_data_object_for_current_frame.get_array(
                    self.transformed_combined_datastructure
                )
                self.input_indices = [
                    self.transformed_combined_datastructure.index(input_type) for input_type in self.input_data_structure
                ]
                self.output_indices = [
                    self.transformed_combined_datastructure.index(input_type) for input_type in self.output_data_structure
                ]
                self.tp_kmp[frame_index].fit(local_data_as_np_array, self.input_indices, self.output_indices)
            else:
                raise NotImplementedError

    def predict(self, x_test: np.ndarray) -> tuple[list[DataObject], np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the mean and covariance over all KMPs.
        :param x_test: x in f(x). For example time.
        :return:
                - mean in format of the training data (quaternion)
                - covariance matrix
                - epistemic part of the covariance matrix
                - aleatoric part of the covariance matrix
        """
        sigma_pred = []
        aleatoric_pred = []
        epistemic_pred = []
        for kmp in self.tp_kmp:
            kmp.update_inputs(x_test)
            sigma_pred.append(kmp.cov_diag())
            epistemic_pred.append(kmp.epistemic_diag())
            aleatoric_pred.append(kmp.aleatoric_diag())

        mu_pred = []  # Prediction in output_data_structure
        for _index, kmp in enumerate(self.tp_kmp):
            mu_pred_quat = DataObject(
                data=kmp.mean().copy(), data_structure=self.output_data_structure, data_type=DataObjectType.ARRAY_2D
            )
            mu_pred_quat.normalize_rotation()
            mu_pred.append(mu_pred_quat)

        return (
            mu_pred,
            np.array(sigma_pred),
            np.array(epistemic_pred),
            np.array(aleatoric_pred),
        )

    def fuse_predictions(
        self,
        pred_frames_obj: DataObject,
        sigma_pred: np.ndarray,
        sigma_pred_aleatoric: np.ndarray,
        sigma_pred_epistemic: np.ndarray,
        mu_pred: list[DataObject],
    ) -> tuple[DataObject, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fusing the predictions of all KMPs (all Frames) together using block-wise operations.
        :param pred_frames_obj: Frames used for the prediction
        :param sigma_pred: diagonal covariance vectors (n_frames, N_test * dim_out)
        :param sigma_pred_aleatoric: aleatoric diagonal vectors
        :param sigma_pred_epistemic: epistemic diagonal vectors
        :param mu_pred: mean of the prediction in quaternion representation
        :return: mean and covariance of the combined prediction
                - mean as DataObject
                - covariance variance array (N_test, dim_out)
                - epistemic variance array (N_test, dim_out)
                - aleatoric variance array (N_test, dim_out)
        """
        if not all(key in pred_frames_obj.get_data_structure() for key in ORIENTATION_QUAT_W_LAST):
            raise NotImplementedError("Only quaternions with w last is supported!")

        if pred_frames_obj.get_shape()[0] != len(mu_pred):
            raise ValueError(
                f"The number of pred frame objects must match the number of predictions, "
                f"but got {pred_frames_obj.get_shape()[0]} pred frame objects and {len(mu_pred)} predictions"
            )

        # Get the rotation matrices and translation vectors
        A_rot_obj, b_obj = self.get_rotation_and_translation(frames_obj_for_prediction=pred_frames_obj)

        # Compute Gaussian product
        if all(key in self.output_data_structure for key in ORIENTATION_QUAT_W_LAST) or all(
            key in self.output_data_structure for key in ORIENTATION_VECTOR_6D
        ):

            dim = self.nb_dim_out
            N = sigma_pred[0].shape[0] // dim

            # Get per-frame rotation matrices (n_frames, dim, dim) and translations (n_frames, dim)
            A_raw = A_rot_obj.get_array(self.output_data_structure)
            b_raw = b_obj.get_array(self.output_data_structure)

            # Accumulators for block-wise precision fusion: (N, dim, dim)
            sum_precision = np.zeros((N, dim, dim))
            sum_precision_al = np.zeros((N, dim, dim))
            sum_precision_ep = np.zeros((N, dim, dim))
            sum_weighted_mu = np.zeros((N, dim, 1))

            idx = np.arange(dim)

            for frame_index in range(len(self.tp_kmp)):
                A = A_raw[frame_index]  # (dim, dim)
                b = b_raw[frame_index]  # (dim,)

                # Reshape diagonal vectors from (N*dim,) to (N, dim)
                s = sigma_pred[frame_index].reshape(N, dim)
                s_al = sigma_pred_aleatoric[frame_index].reshape(N, dim)
                s_ep = sigma_pred_epistemic[frame_index].reshape(N, dim)

                # Build batch (N, dim, dim) diagonal matrices
                S = np.zeros((N, dim, dim))
                S[:, idx, idx] = s
                S_al = np.zeros((N, dim, dim))
                S_al[:, idx, idx] = s_al
                S_ep = np.zeros((N, dim, dim))
                S_ep[:, idx, idx] = s_ep

                # A @ S @ A.T for each point: (1,d,d) @ (N,d,d) @ (1,d,d) = (N,d,d)
                ASAT = A[None] @ S @ A.T[None]
                ASAT_al = A[None] @ S_al @ A.T[None]
                ASAT_ep = A[None] @ S_ep @ A.T[None]

                # Batch dim x dim inversions
                precision = np.linalg.inv(ASAT)
                precision_al = np.linalg.inv(ASAT_al)
                precision_ep = np.linalg.inv(ASAT_ep)

                sum_precision += precision
                sum_precision_al += precision_al
                sum_precision_ep += precision_ep

                # Rotate and translate mean: A @ mu + b for each point
                mu_data = mu_pred[frame_index].get_array(self.output_data_structure)  # (N, dim)
                rotated_mu = (A @ mu_data.T).T + b  # (N, dim)
                sum_weighted_mu += precision @ rotated_mu[:, :, None]  # (N, dim, 1)

            # Final fusion: invert accumulated precision to get covariance
            sigma_fused = np.linalg.inv(sum_precision)  # (N, dim, dim)
            sigma_fused_al = np.linalg.inv(sum_precision_al)
            sigma_fused_ep = np.linalg.inv(sum_precision_ep)

            mu_TP = (sigma_fused @ sum_weighted_mu).squeeze(-1)  # (N, dim)

            mu_TP_obj = DataObject(mu_TP, self.output_data_structure, DataObjectType.ARRAY_2D)
            mu_TP_obj.normalize_rotation()

            # Extract diagonal variance: (N, dim)
            sigma_TP = sigma_fused[:, idx, idx]
            sigma_TP_ep = sigma_fused_ep[:, idx, idx]
            sigma_TP_al = sigma_fused_al[:, idx, idx]

            return mu_TP_obj, sigma_TP, sigma_TP_ep, sigma_TP_al
        else:
            raise NotImplementedError

    def add_placeholder_kmp(self, x_test: np.ndarray):
        """
        Adds a KMP (Frame) which zero as means and a high covariance everywhere as it is not trained on the data.
        It is just used to add via-points to it so those via-points move with this new frame.
        """
        covariance_value = 10000.0

        new_kmp = Kmp(
            gmm_n_components=self.params.k_gmm,
            N=self.params.number_of_test_points,
            length_scale=self.params.length_scale_of_the_kernel,
            h=self.params.proportional_kernel_scaling_factor,
            lambda1=self.params.lambda1,
            lambda2=self.params.lambda2,
            alpha=self.params.alpha,
        )
        new_kmp.mu = np.zeros((self.params.number_of_test_points, len(self.output_indices)))
        new_kmp.sigma = np.full(
            (self.params.number_of_test_points, len(self.output_indices), len(self.output_indices)),
            fill_value=covariance_value,
            dtype=np.float64,
        )
        new_kmp.sigma += np.random.uniform(
            -1.0, 1.0, new_kmp.sigma.shape
        )  # Adding some noise to ensure that inverting of the singular matrix does work
        new_kmp.nb_dim_in = np.size(self.input_indices)
        new_kmp.nb_dim_out = np.size(self.output_indices)
        x_in = np.zeros((self.params.number_of_test_points, new_kmp.nb_dim_in))
        new_kmp.x_in = x_in.reshape(-1, new_kmp.nb_dim_in)
        new_kmp.mu_block = new_kmp.mu.reshape(-1, 1)
        new_kmp.x_test = x_test
        self.tp_kmp.append(new_kmp)

    def get_rotation_and_translation(self, frames_obj_for_prediction: DataObject) -> tuple[DataObject, DataObject]:
        """

        :param frames_obj_for_prediction     : The frames used for calculation the rotation and translation.
        :return:
        """
        if not isinstance(frames_obj_for_prediction, DataObject):
            raise TypeError(f"Input must be a DataObject, not: {type(frames_obj_for_prediction)}")
        if frames_obj_for_prediction.get_data_type() != DataObjectType.ARRAY_2D:
            raise ValueError("This only works for 2D Arrays")

        # one for A traj rot, A traj skew, b traj, the first two are 2D symmetric matrices, the last one is a 2D Array
        nb_dim_out = len(self.output_data_structure)
        number_of_frames = frames_obj_for_prediction.get_shape()[0]

        if not all(ele in self.output_data_structure for ele in POSITION):
            raise NotImplementedError("This function always assume there is a position!")

        b = np.zeros((number_of_frames, nb_dim_out))

        # set the pose at the correct position
        pos_indices = [self.output_data_structure.index(key) for key in POSITION]
        b[:, pos_indices] = frames_obj_for_prediction.get_array(POSITION)

        b_obj = DataObject(b, self.output_data_structure, DataObjectType.ARRAY_2D)

        if all(ele in self.output_data_structure for ele in [*POSITION, *ORIENTATION_QUAT_W_FIRST]):
            A_rotation = np.zeros((number_of_frames, nb_dim_out, nb_dim_out))
            for frame_index in range(number_of_frames):
                # make_skew_matrix_for_w_last and from_quat expect both W_LAST
                quat_w_last: np.ndarray = frames_obj_for_prediction.get_rows([frame_index]).get_array(ORIENTATION_QUAT_W_LAST)[
                    0
                ]

                curr_rot_mat = Rotation.from_quat(quat_w_last).as_matrix()
                # the hardcoded :3, :3 is okay as we use a few lines down POSITION to define these three values
                A_rotation[frame_index, :3, :3] = curr_rot_mat

                # use a w last skew matrix
                # the hardcoded 3:, 3: is okay as we use a few lines down to ORIENTATION_QUAT_W_LAST define these three values
                A_rotation[frame_index, 3:, 3:] = make_skew_matrix_for_w_last(reference_quat_w_last=quat_w_last)

            # Orientation w last is determined by how the skew matrix is constructed
            A_rotation_obj = DataObject(A_rotation, [*POSITION, *ORIENTATION_QUAT_W_LAST], DataObjectType.SYM_CONV_MATRIX)
            return A_rotation_obj, b_obj
        elif all(ele in self.output_data_structure for ele in [*POSITION, *ORIENTATION_VECTOR_6D]):
            # the output is a Position and 6D vector
            A_rotation = np.zeros((number_of_frames, nb_dim_out, nb_dim_out))
            frames_as_vector_6d: np.ndarray = frames_obj_for_prediction.get_rotation_array(RotationOutputType.ROTATION_MATRIX)
            for frame_index in range(frames_obj_for_prediction.get_shape()[0]):
                # the kron maps the rotation matrix three times, one for POSITION and two times for ORIENTATION_VECTOR_6D
                A_rotation[frame_index] = np.kron(np.eye(3), frames_as_vector_6d[frame_index])

            A_rotation_obj = DataObject(A_rotation, [*POSITION, *ORIENTATION_VECTOR_6D], DataObjectType.SYM_CONV_MATRIX)
            return A_rotation_obj, b_obj
        elif len(self.output_data_structure) == 3:  # only xyz is used
            curr_rot_mat = frames_obj_for_prediction.get_rotation_array(RotationOutputType.ROTATION_MATRIX)
            A_rotation = curr_rot_mat

            A_rotation_obj = DataObject(A_rotation, [*POSITION], DataObjectType.SYM_CONV_MATRIX)
            return A_rotation_obj, b_obj
        else:
            raise NotImplementedError

    def add_via_points_locally_to_kmp(
        self, via_local: list[DataObject], kmp_idx: int, time_stamps_for_insertions, variance: float = 1e-8
    ) -> None:
        """
        Adds one or several via points locally at the given time stamps to one of the KMPs by the given index.
        :param via_local: dataobject of the via-point to add
        :param kmp_idx: 0 for the first, 1 for the second frame etc.
        :param time_stamps_for_insertions: List of timesteps where the via-points should be inserted
            (several via-points possible).
        """
        if not isinstance(via_local, list):
            raise TypeError(f"via_local must be a list, got {type(via_local)}")
        if len(via_local) > 0:
            via_local_combined = via_local[0].concatenate_multiple(via_local[1:])
        else:
            via_local_combined = via_local[0]
        output_types = [self.transformed_combined_datastructure[index] for index in self.output_indices]
        self.tp_kmp[kmp_idx].add_viapoints(
            input_via=time_stamps_for_insertions, output_via=via_local_combined.get_array(output_types), gamma=variance
        )

        # For plotting
        if kmp_idx not in self.local_via_points_per_frame:
            self.local_via_points_per_frame[kmp_idx] = []

        for via_point_obj in via_local:
            self.local_via_points_per_frame[kmp_idx].append(via_point_obj)

    def get_variance_of_covariance(self, mu_TP: DataObject, mu_variance: np.ndarray) -> np.ndarray | None:
        """
        Returns the variance (diagonal of a given covariance matrix) and mean values.
        Supports multiple input formats:
          - (N*dim,) diagonal vector from _diag methods
          - (N, dim) already-extracted variance array from block-wise fusion
          - (N*dim, N*dim) full covariance matrix (legacy)
        :param mu_TP: mean values as DataObject
        :param mu_variance: covariance in any of the supported formats
        :return: np.ndarray of shape (N, dim) with variance for each mean value
        """
        N = mu_TP.get_shape()[0]
        dim = self.nb_dim_out

        variance_predicted_points: np.ndarray | None
        if all(ele in self.output_data_structure for ele in ORIENTATION_SO3_MANIFOLD):
            variance_predicted_points = None
        elif mu_variance.ndim == 1:
            # Diagonal vector (N*dim,) from _diag methods
            variance_predicted_points = mu_variance.reshape(N, dim)
        elif mu_variance.ndim == 2 and mu_variance.shape == (N, dim):
            # Already (N, dim) format from block-wise fusion
            variance_predicted_points = mu_variance
        else:
            # Legacy: full (N*dim, N*dim) covariance matrix
            variance_predicted_points = np.zeros(mu_TP.get_shape())
            for j in range(N):
                start, end = j * dim, (j + 1) * dim
                variance_predicted_points[j] = np.diag(mu_variance[start:end, start:end])

        return variance_predicted_points

    def get_std_of_covariance(self, mu_TP: DataObject, mu_variance: np.ndarray) -> np.ndarray | None:
        """
        Returns the std of the diagonal (=variance) of a given covariance matrix and mean values
        :param mu_TP: mean values as DataObject
        :param mu_variance: covariance matrix for the given mean values
        :return: np.ndarray of the std for each mean value of the prediction
        """
        if all(ele in self.output_data_structure for ele in ORIENTATION_SO3_MANIFOLD):
            std_predicted_points = None
        else:
            variance_predicted_points = self.get_variance_of_covariance(mu_TP=mu_TP, mu_variance=mu_variance)
            assert variance_predicted_points is not None
            std_predicted_points = np.sqrt(np.maximum(variance_predicted_points, 0.0))

        return std_predicted_points

    def add_via_point_to_placeholder_frame(
        self, times: list[float], viapoint_position: list[float], new_frame_obj: DataObject, frame_index
    ):
        """Add via-points to a placeholder frame, transforming them into the frame's local coordinates."""
        via_point_list = []
        for timestamp in times:
            via_point_list.append(
                DataObject(
                    data=np.array([[timestamp, *viapoint_position, frame_index]]),
                    data_structure=[InputType.T, *POSITION, *ORIENTATION_QUAT_W_FIRST, InputType.FRAME_INDEX],
                    data_type=DataObjectType.ARRAY_2D,
                )
            )
        via_points_list_mapped_locally = []
        for via_point in via_point_list:
            via_points_list_mapped_locally.append(via_point.map_to_different_data_object(new_frame_obj))
        self.add_via_points_locally_to_kmp(
            kmp_idx=frame_index, time_stamps_for_insertions=times, via_local=via_points_list_mapped_locally, variance=1e-8
        )
        # For plotting
        if frame_index not in self.local_via_points_per_frame:
            self.local_via_points_per_frame[frame_index] = []

        for via_point_obj in via_point_list:
            self.global_via_points.append(via_point_obj)
            self.local_via_points_per_frame[frame_index].append(via_point_obj.map_to_different_data_object(new_frame_obj))

    def plot_demonstrations(  # noqa: C901
        self,
        mu_local: list[DataObject],
        sigma_pred_local: np.ndarray,
        mu_global: DataObject,
        variance_global: np.ndarray,
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
        show_plot: bool = True,
    ):
        """
        Default for all figures, shows all recorded demonstrations globally including mean, covar

        """
        sns.set_theme()

        demonstrations_global = []
        demonstrations_local_frame0 = []
        demonstrations_local_frame1 = []
        for dh in self.data_handler:
            demonstrations_global.append(dh.get_data().get_array(dh.get_data_structure()))
            demonstrations_local_frame0.append(dh.get_transformed_demonstration_for_frames_index(0))
            demonstrations_local_frame1.append(dh.get_transformed_demonstration_for_frames_index(1))

        demonstrations_local: list[Any] = [np.array(demonstrations_local_frame0), np.array(demonstrations_local_frame1)]
        demonstrations_global_arr: np.ndarray = np.array(demonstrations_global)

        if show_frame_plots and show_frame_3:
            raise ValueError("Only one of them can be true at once")

        titles = ["Box 1 frame", "Box 2 frame", "Global frame"]

        global_std = self.get_std_of_covariance(mu_global, variance_global)
        local_std = [
            self.get_std_of_covariance(mu_TP=mu_pred_local, mu_variance=sigma_pred_local[i])
            for i, mu_pred_local in enumerate(mu_local)
        ]

        all_demonstrations: list[tuple[Any, ...]]
        if show_frame_plots:
            all_demonstrations = [
                (demonstrations_local[0], mu_local[0], local_std[0], None, self.local_via_points_per_frame.get(0, []), None),
                (demonstrations_local[1], mu_local[1], local_std[1], None, self.local_via_points_per_frame.get(1, []), None),
            ]
            for extra_idx in range(2, len(mu_local)):
                all_demonstrations.append(
                    (
                        None,
                        mu_local[extra_idx],
                        local_std[extra_idx],
                        None,
                        self.local_via_points_per_frame.get(extra_idx, []),
                        None,
                    )
                )
                titles.insert(-1, "Camera frame")
            all_demonstrations.append((demonstrations_global_arr, mu_global, global_std, None, self.global_via_points, None))
        elif show_frame_3:
            all_demonstrations = [
                (None, mu_local[2], local_std[2], None, self.local_via_points_per_frame.get(2, []), None),
                (demonstrations_global_arr, mu_global, global_std, None, self.global_via_points, None),
            ]
            titles = ["Camera frame", "Global frame"]

        else:
            all_demonstrations = [(demonstrations_global_arr, mu_global, global_std, None, self.global_via_points, None)]

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
                    # Handle DataObject arrays - extract actual data
                    if len(current_demonstrations.shape) == 1:
                        # Array of DataObject instances - extract data from each
                        demo_arrays_list = []
                        for demo_obj in current_demonstrations:
                            if hasattr(demo_obj, "get_array"):
                                demo_data = demo_obj.get_array(demo_obj.get_data_structure())
                                demo_arrays_list.append(demo_data)

                        if demo_arrays_list:
                            demo_arrays = np.array(demo_arrays_list)

                            # Now plot the extracted data
                            for x in range(demo_arrays.shape[0]):
                                plt.plot(
                                    demo_arrays[x, :, 0],
                                    demo_arrays[x, :, i + 1],
                                    color=black_color,
                                    linestyle=":",
                                    linewidth=1,
                                    label="Demonstrations" if x == 0 else None,
                                )
                            min_value = np.min([min_value, np.min(demo_arrays[:, :, i + 1])])
                            max_value = np.max([max_value, np.max(demo_arrays[:, :, i + 1])])
                    elif len(current_demonstrations.shape) == 2:
                        # If 2D, treat as single demonstration
                        plt.plot(
                            current_demonstrations[:, 0],
                            current_demonstrations[:, i + 1],
                            color=black_color,
                            linestyle=":",
                            linewidth=1,
                            label="Demonstrations",
                        )
                        min_value = np.min([min_value, np.min(current_demonstrations[:, i + 1])])
                        max_value = np.max([max_value, np.max(current_demonstrations[:, i + 1])])
                    else:
                        # If 3D, iterate over demonstrations
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

                # Extract data from DataObjects if needed
                if hasattr(current_mu, "get_array"):
                    mu_data = current_mu.get_array(current_mu.get_data_structure())
                else:
                    mu_data = current_mu

                if hasattr(current_std, "get_array"):
                    std_data = current_std.get_array(current_std.get_data_structure())
                else:
                    std_data = current_std

                # Generate x-axis based on actual prediction length
                x_axis = (
                    np.linspace(0, 1, len(mu_data)) if len(mu_data) != len(self.tp_kmp[0].x_in) else self.tp_kmp[0].x_in[:, 0]
                )

                if current_robot_movement is None:
                    # plot mu - we only show mu if we don't plot robot movement.
                    plt.plot(x_axis, mu_data[:, i], color=red_color, label="Prediction")

                # plot variance
                number_of_legend_cols = 2
                if current_std_epistemic is None:
                    plt.fill_between(
                        x_axis,
                        mu_data[:, i] - std_data[:, i] * scale_std_factor,
                        mu_data[:, i] + std_data[:, i] * scale_std_factor,
                        color=red_color,
                        alpha=0.2,
                        label="Std. Dev.",
                    )
                    std_data[:, i][np.isnan(std_data[:, i])] = 0.0
                    min_value = np.min([min_value, np.min(mu_data[:, i] + std_data[:, i] * scale_std_factor)])
                    min_value = np.min([min_value, np.min(mu_data[:, i] - std_data[:, i] * scale_std_factor)])
                    max_value = np.max([max_value, np.max(mu_data[:, i] + std_data[:, i] * scale_std_factor)])
                    max_value = np.max([max_value, np.max(mu_data[:, i] - std_data[:, i] * scale_std_factor)])
                else:
                    current_std_epistemic[np.isnan(current_std_epistemic)] = 0.0
                    aleatoric_std = std_data - np.abs(current_std_epistemic)

                    plt.fill_between(
                        x_axis,
                        mu_data[:, i] - aleatoric_std[:, i] * scale_std_factor,
                        mu_data[:, i] + aleatoric_std[:, i] * scale_std_factor,
                        color=green_color,
                        alpha=0.2,
                        label="Aleatoric",
                    )
                    plt.fill_between(
                        x_axis,
                        mu_data[:, i] - current_std_epistemic[:, i] * scale_std_factor,
                        mu_data[:, i] + current_std_epistemic[:, i] * scale_std_factor,
                        color=blue_color,
                        alpha=0.2,
                        label="Epistemic",
                    )

                    aleatoric_std[:, i][np.isnan(aleatoric_std[:, i])] = 0.0
                    min_value = np.min([min_value, np.min(mu_data[:, i] + aleatoric_std[:, i] * scale_std_factor)])
                    min_value = np.min([min_value, np.min(mu_data[:, i] - aleatoric_std[:, i] * scale_std_factor)])
                    max_value = np.max([max_value, np.max(mu_data[:, i] + aleatoric_std[:, i] * scale_std_factor)])
                    max_value = np.max([max_value, np.max(mu_data[:, i] - aleatoric_std[:, i] * scale_std_factor)])
                    min_value = np.min([min_value, np.min(mu_data[:, i] + current_std_epistemic[:, i] * scale_std_factor)])
                    min_value = np.min([min_value, np.min(mu_data[:, i] - current_std_epistemic[:, i] * scale_std_factor)])
                    max_value = np.max([max_value, np.max(mu_data[:, i] + current_std_epistemic[:, i] * scale_std_factor)])
                    max_value = np.max([max_value, np.max(mu_data[:, i] - current_std_epistemic[:, i] * scale_std_factor)])
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
                    plt.plot(x_axis, current_robot_movement[:, i], color=red_color, label="Robot movement")

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
            for _j in range(2):
                ax[_j].set_xticklabels([])
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
        if show_plot:
            plt.show()
