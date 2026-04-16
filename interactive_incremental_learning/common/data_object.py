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
Script to manage different data types, e.g. robot recordings.

If you are using this code please cite us:
M. Knauer, A. Albu-Schäffer, F. Stulp and J. Silvério, "Interactive Incremental Learning of Generalizable
Skills With Local Trajectory Modulation," in IEEE Robotics and Automation Letters (RA-L), vol. 10, no. 4,
pp. 3398-3405, April 2025, doi: 10.1109/LRA.2025.3542209

See CITATION.bib for the bib file!
"""

from __future__ import annotations

__author__ = "Markus Knauer, Joao Silverio"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

from enum import Enum
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from interactive_incremental_learning.common.tp_math import make_skew_matrix_for_w_first, make_skew_matrix_for_w_last


class InputType(Enum):
    T = "t"  # Time
    X = "x"  # Positions
    Y = "y"
    Z = "z"
    W = "w"  # Quaternions
    WX = "wx"
    WY = "wy"
    WZ = "wz"
    VX = "vx"
    VY = "vy"  # 6D Vector Orientation representation
    VZ = "vz"
    UX = "ux"
    UY = "uy"
    UZ = "uz"
    T1 = "t1"  # SO3 Manifold Tangentspace
    T2 = "t2"
    T3 = "t3"
    FRAME_INDEX = "frame_index"

    def __repr__(self) -> str:
        return self.value


POSITION: list[InputType] = [InputType.X, InputType.Y, InputType.Z]
ORIENTATION_QUAT_W_LAST: list[InputType] = [InputType.WX, InputType.WY, InputType.WZ, InputType.W]
ORIENTATION_QUAT_W_FIRST: list[InputType] = [InputType.W, InputType.WX, InputType.WY, InputType.WZ]
ORIENTATION_VECTOR_6D: list[InputType] = [InputType.VX, InputType.VY, InputType.VZ, InputType.UX, InputType.UY, InputType.UZ]
ORIENTATION_VECTOR_6D_FIRST: list[InputType] = [InputType.VX, InputType.VY, InputType.VZ]
ORIENTATION_VECTOR_6D_SECOND: list[InputType] = [InputType.UX, InputType.UY, InputType.UZ]
ORIENTATION_SO3_MANIFOLD: list[InputType] = [InputType.T1, InputType.T2, InputType.T3]


class DataObjectType(Enum):
    ARRAY_2D = 0
    SYM_CONV_MATRIX = 1
    UNSYM_CONV_MATRIX = 2
    UNKNOWN = 3

    def is_conv_matrix(self) -> bool:
        return self.value in [DataObjectType.SYM_CONV_MATRIX, DataObjectType.UNSYM_CONV_MATRIX]


class RotationOutputType(Enum):
    QUATERNION_W_FIRST = 0  # here the order is wx, wy, wz, w
    QUATERNION_W_LAST = 1  # here the order is w, wx, wy, wz
    ROTATION_MATRIX = 2
    VECTOR_6D = 3
    SKEW_MATRIX_W_FIRST = 4
    SKEW_MATRIX_W_LAST = 5
    SO3_MANIFOLD = 6
    UNKNOWN = 7

    def to_input_types(self) -> list[InputType]:
        return {
            RotationOutputType.QUATERNION_W_FIRST: ORIENTATION_QUAT_W_FIRST,
            RotationOutputType.QUATERNION_W_LAST: ORIENTATION_QUAT_W_LAST,
            RotationOutputType.VECTOR_6D: ORIENTATION_VECTOR_6D,
            RotationOutputType.SO3_MANIFOLD: ORIENTATION_SO3_MANIFOLD,
        }[self]

    @staticmethod
    def from_input_types(input_types: list[InputType]) -> RotationOutputType:
        for rot_type in [RotationOutputType.QUATERNION_W_FIRST, RotationOutputType.SO3_MANIFOLD, RotationOutputType.VECTOR_6D]:
            if all(input_type in input_types for input_type in rot_type.to_input_types()):
                return rot_type
        return RotationOutputType.UNKNOWN


def convert_to_rotation_matrix(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Two 3D vectors v and u are converted to a 3x3 rotation matrix
    """

    v_norm = v / np.linalg.norm(v)
    u_norm = u / np.linalg.norm(u - np.dot(v_norm, u) * v_norm)

    w = np.cross(v_norm, u_norm)

    return np.stack([v_norm, u_norm, w], axis=0)


DataStructureType = list[InputType] | dict[str, list[InputType]]


def is_list_of_input_types(value: DataStructureType) -> bool:
    return isinstance(value, list) and all(isinstance(ele, InputType) for ele in value)


def is_dict_of_list_of_input_types(value: DataStructureType) -> bool:
    return (
        isinstance(value, dict)
        and all(isinstance(key, str) and key in ["rows", "cols"] for key in value.keys())
        and all(all(isinstance(ele, InputType) for ele in current_row) for current_row in value.items())
    )


def assert_list_of_input_types(value: Any) -> list[InputType]:
    if isinstance(value, list):
        if not all(isinstance(ele, InputType) for ele in value):
            raise TypeError(
                f"If a list is given all values must be InputType: {value}, " f"{[type(e) is InputType for e in value]}"
            )
        return value
    else:
        raise TypeError(f"The type does not match: {value}, desired type: {list[InputType]}")


def assert_dict_of_list_of_input_types(value: Any) -> dict[str, list[InputType]]:
    if isinstance(value, dict):
        if not all(isinstance(key, str) and key in ["rows", "cols"] for key in value.keys()):
            raise TypeError(f'All keys must be either "rows" or "cols": {value.keys()}')
        if not all(all(isinstance(ele, InputType) for ele in current_row) for current_row in value.items()):
            raise TypeError(f"All values inside the rows/cols must be InputType: {value}")
        return value
    else:
        raise TypeError(f"The type does not match: {value}, desired type: {dict[str, list[InputType]]}")


def assert_is_data_structure_type(value: Any) -> DataStructureType:
    if isinstance(value, list):
        if not all(isinstance(ele, InputType) for ele in value):
            raise TypeError(f"If a list is given all values must be InputType: {value}")
        return value
    elif isinstance(value, dict):
        if not all(isinstance(key, str) and key in ["rows", "cols"] for key in value.keys()):
            raise TypeError(f'All keys must be either "rows" or "cols": {value.keys()}')
        if not all(all(isinstance(ele, InputType) for ele in current_row) for current_row in value.items()):
            raise TypeError(f"All values inside the rows/cols must be InputType: {value}")
        return value
    else:
        raise TypeError(f"The type does not match: {value}, desired type: {DataStructureType}")


class DataObject:
    def __init__(
        self,
        data: np.ndarray,
        data_structure: DataStructureType,
        data_type: DataObjectType,
    ):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Input must be a np.ndarray, not: {type(data)}")

        self._data = data

        if not isinstance(data_type, DataObjectType):
            raise ValueError(f"The data type is wrong: {data_type}")
        self._data_type = data_type
        if self._data_type.is_conv_matrix() and len(self._data.shape) != 3:
            raise ValueError("The data must have three dimensions if the data is a matrix!")

        data_structure = assert_is_data_structure_type(data_structure)

        self._data_structure_linear_mapping: dict[InputType, int] = {}
        self._data_structure_two_dim_mapping: dict[str, dict[InputType, int]] = {}

        if self._data_type in [DataObjectType.UNSYM_CONV_MATRIX]:
            data_structure = assert_dict_of_list_of_input_types(data_structure)
            self._data_structure_two_dim_mapping = {
                "rows": {key: index for index, key in enumerate(data_structure["rows"])},
                "cols": {key: index for index, key in enumerate(data_structure["cols"])},
            }
            if len(self._data_structure_two_dim_mapping["rows"]) != self._data.shape[1]:
                raise ValueError(
                    f"The data ({self._data.shape}) has the wrong amount of rows {self._data.shape[1]}, "
                    f"to fit the described data structure {self._data_structure_two_dim_mapping['rows']}."
                )
            if len(self._data_structure_two_dim_mapping["cols"]) != self._data.shape[2]:
                raise ValueError(
                    f"The data ({self._data.shape}) has the wrong amount of cols {self._data.shape[2]}, "
                    f"to fit the described data structure {self._data_structure_two_dim_mapping['cols']}."
                )
        else:
            data_structure = assert_list_of_input_types(data_structure)
            self._data_structure_linear_mapping = {key: index for index, key in enumerate(data_structure)}
            if self._data_type == DataObjectType.ARRAY_2D:
                self._data_structure_linear_mapping = {key: index for index, key in enumerate(data_structure)}
                if len(self._data_structure_linear_mapping) != self._data.shape[1]:
                    raise ValueError(
                        f"The data has as shape {self._data.shape[1]}, but there are {len(data_structure)} "
                        f"elements in {data_structure}"
                    )
            elif self._data_type == DataObjectType.SYM_CONV_MATRIX:
                if (
                    len(self._data_structure_linear_mapping) != self._data.shape[1]
                    or len(self._data_structure_linear_mapping) != self._data.shape[2]
                ):
                    raise ValueError(
                        f"The data has as shape {self._data.shape[1:3]}, but there are {len(data_structure)} "
                        f"elements in {data_structure}"
                    )
            else:
                raise ValueError(f"This data type is unknown here: {self._data_type}")

    def map_to_different_data_object(self, reference_frame: DataObject) -> DataObject:
        """
        Returns a DataObject containing the requested data structure and data_type
        :param reference_frame: DataObject which contains only a single row and this row should have the elements
                                [x, y, z, w, wx, wy, wz]
        :return: DataObject mapped to the new space
        """
        if self.get_data_type() != DataObjectType.ARRAY_2D:
            raise ValueError("This only works for ARRAY_2D DataObjects")

        new_data_object = DataObject(
            self._data.copy(), data_structure=self.get_data_structure(), data_type=DataObjectType.ARRAY_2D
        )
        new_data_object._map_points_into_frame_coordinate_system(reference_frame=reference_frame)

        return new_data_object

    def get_rows(self, row_indices: np.ndarray | list[int]) -> DataObject:
        """
        Get the rows of the DataObject as new DataObject, if only one row is requested a list with just one element
        also works.

        A row in a ARRAY_2D could corresponds to a POSITION combined with QUATERNION.

        :param row_indices: List of row indices
        :return:
        """
        if self.get_data_type() != DataObjectType.ARRAY_2D:
            raise RuntimeError("This only works on 2D ARRAYs")
        new_data = self._data.copy()[row_indices]

        new_data_object = DataObject(
            data=new_data, data_structure=self.get_data_structure(), data_type=DataObjectType.ARRAY_2D
        )
        return new_data_object

    def get_shape(self) -> tuple[int, ...]:
        return self._data.shape

    def get_array(
        self,
        request_data_structure: DataStructureType,
        keep_shape: bool = True,
    ) -> np.ndarray:
        """
        Get a np.ndarray of the requested data.
        :param request_data_structure: e. g. [x, y, z] if you want to have position
        :param keep_shape:
        :return: np.ndarray with requested data structure and data_type
        """
        request_data_structure = assert_is_data_structure_type(request_data_structure)

        if isinstance(request_data_structure, list):
            request_data_structure = assert_list_of_input_types(request_data_structure)
            req_indices = [self._get_index(req) for req in request_data_structure]
            if self._data_type == DataObjectType.ARRAY_2D:
                return self._data[:, req_indices]
            elif self._data_type == DataObjectType.SYM_CONV_MATRIX:
                return self._data[:, req_indices][:, :, req_indices]
            else:
                raise ValueError(f"The requested list data structure does not work with this type: {self._data_type}")
        elif is_dict_of_list_of_input_types(request_data_structure):
            if self._data_type == DataObjectType.UNSYM_CONV_MATRIX:
                if "rows" not in request_data_structure or "cols" not in request_data_structure:
                    raise ValueError(f"The rows and cols keys have to be defined for {request_data_structure.keys()}")
                req_indices_rows = [self._get_index(req, "rows") for req in request_data_structure["rows"]]
                req_indices_cols = [self._get_index(req, "cols") for req in request_data_structure["cols"]]
                if keep_shape:
                    if len(req_indices_cols) == 1 and len(req_indices_rows) == 1:
                        return self._data[:, req_indices_rows[0] : req_indices_rows[0] + 1][
                            :, :, req_indices_cols[0] : req_indices_cols[0] + 1
                        ]
                    elif len(req_indices_cols) == 1 and len(req_indices_rows) > 1:
                        return self._data[:, req_indices_rows][:, :, req_indices_cols[0] : req_indices_cols[0] + 1]
                    elif len(req_indices_cols) > 1 and len(req_indices_rows) == 1:
                        return self._data[:, req_indices_rows[0] : req_indices_rows[0] + 1][:, :, req_indices_cols]
                    else:
                        return self._data[:, req_indices_rows][:, :, req_indices_cols]
                else:
                    return self._data[:, req_indices_rows, req_indices_cols]
            else:
                raise ValueError(f"The requested dict data structure does not work with this type: {self._data_type}")
        else:
            raise NotImplementedError(f"This data structure is not yet supported here: {request_data_structure}")

    def _get_index(self, requested_data_value: InputType, used_axis: str | None = None) -> int:
        """
        Return index of a given data_value. e.g. If you want to know the index of x inside the DataObject.
        :param requested_data_value: e.g. InputType.X for position x.
        :param used_axis: In case of a UNSYM CONV MATRIX
        :return: int value representing the index of the requested data value inside the DataObject.
        """
        if self._data_type == DataObjectType.UNSYM_CONV_MATRIX:
            if used_axis is None:
                raise ValueError("The used axis has to be set for the UNSYM CONV MATRIX")
            if requested_data_value not in self._data_structure_two_dim_mapping[used_axis].keys():
                raise ValueError(
                    f"The requested data value: {requested_data_value} is not in the data "
                    f"structure: {self._data_structure_two_dim_mapping[used_axis]}, used axis is {used_axis}."
                )
            return self._data_structure_two_dim_mapping[used_axis][requested_data_value]
        elif self._data_type in [DataObjectType.ARRAY_2D, DataObjectType.SYM_CONV_MATRIX]:
            if requested_data_value not in self._data_structure_linear_mapping.keys():
                raise ValueError(
                    f"The requested data value: {requested_data_value} is not in the data "
                    f"structure: {self._data_structure_linear_mapping}"
                )
            return self._data_structure_linear_mapping[requested_data_value]
        else:
            raise ValueError(f"This data type is unknown here: {self._data_type}")

    def get_data_structure(self) -> DataStructureType:
        """
        Returns the data structure
        :return:
        """
        if self._data_type in [DataObjectType.SYM_CONV_MATRIX, DataObjectType.ARRAY_2D]:
            return list(self._data_structure_linear_mapping.keys())
        elif self._data_type in [DataObjectType.UNSYM_CONV_MATRIX]:
            return {
                "rows": list(self._data_structure_two_dim_mapping["rows"].keys()),
                "cols": list(self._data_structure_two_dim_mapping["cols"].keys()),
            }
        else:
            raise NotImplementedError

    def get_data_type(self) -> DataObjectType:
        """
        Returns the data type
        :return:
        """
        return self._data_type

    def _map_points_into_frame_coordinate_system(self, reference_frame: DataObject):
        """
        Maps the points and rotation stored in this DataObject to the given reference frame DataObject
        Both need to have x,y,z and w,wx,wy,wz quats, will change the current DataObject.
        :param reference_frame: DataObject
        """
        if reference_frame.get_shape()[0] != 1:
            raise RuntimeError(
                "The given reference frame has more than one frame coordinate! Use get_rows to reduce the number!"
            )
        if self.get_data_type() != DataObjectType.ARRAY_2D:
            raise RuntimeError("This only works on ARRAY 2D!")

        # scipy.spatial.transform.Rotation expects quaternion in scalar-last (x, y, z, w) format
        rotation_mat = reference_frame.get_rotation_array(RotationOutputType.ROTATION_MATRIX)[0]

        xyz_indices = [self._get_index(letter) for letter in POSITION]
        self._data[:, xyz_indices] -= reference_frame.get_array(POSITION)[0]
        # equal to rotation_mat.T @ self._data[:, xyz_indices]
        self._data[:, xyz_indices] = self._data[:, xyz_indices] @ rotation_mat

        rotation_output_type = self.get_rotation_type()
        if (
            rotation_output_type == RotationOutputType.QUATERNION_W_LAST
            or rotation_output_type == RotationOutputType.QUATERNION_W_FIRST
        ):
            reference_quat = reference_frame.get_rotation_array(RotationOutputType.QUATERNION_W_LAST)[0]
            quat_list = ORIENTATION_QUAT_W_LAST
            quat_indices = [self._get_index(letter) for letter in quat_list]

            # invert the reference quat
            reference_quat[:3] *= -1.0
            skew_matrix = make_skew_matrix_for_w_last(reference_quat)
            self._data[:, quat_indices] = self._data[:, quat_indices] @ skew_matrix.T

        elif rotation_output_type == RotationOutputType.VECTOR_6D:
            rotation_data = self.get_rotation_array(RotationOutputType.ROTATION_MATRIX)

            rotation_data = np.matmul(rotation_data, rotation_mat.T)
            v_list = [self._get_index(letter) for letter in [InputType.VX, InputType.VY, InputType.VZ]]
            u_list = [self._get_index(letter) for letter in [InputType.UX, InputType.UY, InputType.UZ]]

            self._data[:, v_list] = rotation_data[:, 0]
            self._data[:, u_list] = rotation_data[:, 1]
        elif rotation_output_type != RotationOutputType.UNKNOWN:
            raise ValueError("This rotation type is not treated here!")

    def as_affine_transformation_matrices(self) -> np.ndarray:
        """
        Returns the data as [N, 4, 4], where each [4, 4] matrix is an affine transformation with a rotation and a
        translation, which works on homogenous coordinates:
         [R T]
         [0 1]
        :return: list of affine transformations
        """
        new_data = np.empty((self.get_shape()[0], 4, 4))
        new_data[:, :3, :3] = self.get_rotation_array(RotationOutputType.ROTATION_MATRIX)
        new_data[:, :3, 3] = self.get_array(POSITION)
        new_data[:, 3, 3] = 1.0
        return new_data

    def get_rotation_type(self) -> RotationOutputType:
        if self._data_type != DataObjectType.ARRAY_2D:
            raise ValueError(f"This only works for ARRAY_2D, not {self._data_type}")

        data_structure = assert_list_of_input_types(self.get_data_structure())
        if all([letter in data_structure for letter in ORIENTATION_QUAT_W_FIRST]):
            return RotationOutputType.QUATERNION_W_FIRST
        elif all([letter in data_structure for letter in ORIENTATION_VECTOR_6D]):
            return RotationOutputType.VECTOR_6D
        elif all([letter in data_structure for letter in ORIENTATION_SO3_MANIFOLD]):
            return RotationOutputType.SO3_MANIFOLD
        return RotationOutputType.UNKNOWN

    def get_rotation_array(self, desired_output: RotationOutputType) -> np.ndarray:
        if not isinstance(desired_output, RotationOutputType):
            raise TypeError(f"The input can only be a RotationOutputType: {desired_output}")

        rotation_output_type = self.get_rotation_type()
        if rotation_output_type in [RotationOutputType.QUATERNION_W_LAST, RotationOutputType.QUATERNION_W_FIRST]:
            # data is stored as quaternion
            quats = self.get_array(ORIENTATION_QUAT_W_LAST)
            rotations = Rotation.from_quat(quats)
        elif rotation_output_type == RotationOutputType.VECTOR_6D:
            vectors_6d = self.get_array(ORIENTATION_VECTOR_6D)
            rotations = Rotation.from_matrix([convert_to_rotation_matrix(v[:3], v[3:]) for v in vectors_6d])
        elif rotation_output_type == RotationOutputType.SO3_MANIFOLD:
            so3_data = self.get_array(ORIENTATION_SO3_MANIFOLD)
            if desired_output != RotationOutputType.SO3_MANIFOLD:
                raise NotImplementedError("Mapping from SO3 to other rotation representations is not yet supported")
            return so3_data
        elif rotation_output_type == RotationOutputType.UNKNOWN:
            return np.array([])  # no rotation given!
        else:
            raise RuntimeError(
                f"The data {self.get_data_structure()} does not contain rotation information: {rotation_output_type}"
            )

        if desired_output == RotationOutputType.ROTATION_MATRIX:
            return rotations.as_matrix()
        elif desired_output == RotationOutputType.QUATERNION_W_LAST:
            return rotations.as_quat()
        elif desired_output == RotationOutputType.QUATERNION_W_FIRST:
            rot_quats = rotations.as_quat()
            # move w to first position
            return np.concatenate([rot_quats[:, 3:], rot_quats[:, :3]], axis=1)
        elif desired_output == RotationOutputType.SKEW_MATRIX_W_FIRST:
            quat_list = ORIENTATION_QUAT_W_FIRST
            quat_indices = [self._get_index(letter) for letter in quat_list]

            return np.array([make_skew_matrix_for_w_first(quat) for quat in self._data[:, quat_indices]])
        elif desired_output == RotationOutputType.VECTOR_6D:
            rot_mats = rotations.as_matrix()
            return np.concatenate([rot_mats[:, 0], rot_mats[:, 1]], axis=1)
        elif desired_output == RotationOutputType.SO3_MANIFOLD:
            raise NotImplementedError("Conversion to SO3 manifold representation is not yet supported")
        else:
            raise ValueError(f"Unknown desired rotation output type: {desired_output}")

    def set_quaternions_positive(self):
        """
        In order to prevent an overflow of the quaternions, they have to be positive if there is a flip
        from negative to positive and the other way around.
        :param data: np.array of demonstrations in form NrDemos X NrOfPoints X NrOfVariables (t, x, y, z + quaternions)
        :return: sign corrected np.array of demonstrations (same format as "data" above).
        """
        if self.get_rotation_type() not in [RotationOutputType.QUATERNION_W_LAST, RotationOutputType.QUATERNION_W_FIRST]:
            return
        if self.get_data_type() != DataObjectType.ARRAY_2D:
            raise RuntimeError("The data structure must be an ARRAY 2D!")
        quat_indices = [self._get_index(letter) for letter in ORIENTATION_QUAT_W_FIRST]
        for point_index in range(1, len(self._data)):
            dist_pos = 0
            dist_neg = 0
            for i in quat_indices:
                e = self._data[point_index, i]
                old_e = self._data[point_index - 1, i]
                dist_pos += abs(e - old_e)
                dist_neg += abs(-e - old_e)
            if dist_pos >= dist_neg:
                self._data[point_index, quat_indices] *= -1
        # flip everything to positive
        if np.mean(self._data[:, quat_indices[0]]) < 0:
            self._data[:, quat_indices] = -self._data[:, quat_indices]

    def add_fixed_quaternion_to_all_points(self, fixed_quaternion: np.ndarray, data_structure: list[InputType]) -> None:
        """
        Add a fixed quaternion with a given data structure to each row of the DataObject

        :param fixed_quaternion: A fixed quaternion in the form of the datastructure list
        :param data_structure: The datastructure list, e.g. ["w", "wx", "wy", "wz"]
        """
        if self._data_type != DataObjectType.ARRAY_2D:
            raise RuntimeError("This only works for array 2D!")

        if any(key in self._data_structure_linear_mapping for key in data_structure):
            raise ValueError(f"One of the keys in: {data_structure} is already in {self._data_structure_linear_mapping}!")

        # add the new keys to the data structure
        self._data_structure_linear_mapping.update(
            {key: index + len(self._data_structure_linear_mapping) for index, key in enumerate(data_structure)}
        )

        # extend the data array to contain quaternions
        self._data = np.concatenate([self._data, np.zeros((self._data.shape[0], len(data_structure)))], axis=1)

        # iterate over each row and add the quaternions
        used_indices = [self._get_index(val) for val in data_structure]
        self._data[:, used_indices] = fixed_quaternion

    def copy(self) -> DataObject:
        """
        Returns a copy of the DataObject (similar to np.copy())
        """
        return DataObject(
            self.get_array(self.get_data_structure()).copy(),
            data_structure=self.get_data_structure(),
            data_type=self.get_data_type(),
        )

    def concatenate_multiple(self, data_objects: list[DataObject]) -> DataObject:
        """
        Merge a list of data objects with the current data object, all data objects need to have the same data elements
        as the first one.

        :param data_objects: List of DataObjects
        :return: Combined DataObject
        """
        if self.get_data_type() != DataObjectType.ARRAY_2D:
            raise ValueError("concatenate_multiple only works on ARRAY_2D DataObjects")
        for data_object in data_objects:
            if data_object.get_data_type() != DataObjectType.ARRAY_2D:
                raise ValueError("All DataObjects to concatenate must be ARRAY_2D")

        current_datastructure = assert_list_of_input_types(self.get_data_structure())
        external_np_data = [data_object.get_array(current_datastructure) for data_object in data_objects]
        combined_data = np.concatenate([self.get_array(current_datastructure), *external_np_data], axis=0)

        return DataObject(combined_data, current_datastructure, self.get_data_type())

    def normalize_rotation(self):
        if self._data_type != DataObjectType.ARRAY_2D:
            raise ValueError("normalize_rotation only works on ARRAY_2D DataObjects")

        if self.get_rotation_type() == RotationOutputType.UNKNOWN:
            return
        used_rotation_list = []
        if self.get_rotation_type() in [RotationOutputType.QUATERNION_W_FIRST, RotationOutputType.QUATERNION_W_LAST]:
            used_rotation_list.append(ORIENTATION_QUAT_W_FIRST)
        elif self.get_rotation_type() == RotationOutputType.VECTOR_6D:
            used_rotation_list.extend([ORIENTATION_VECTOR_6D_FIRST, ORIENTATION_VECTOR_6D_SECOND])
        else:
            raise NotImplementedError("This rotation type does not have a defined normalization")

        for rotation_list in used_rotation_list:
            used_indices = [self._get_index(key) for key in rotation_list]
            self._data[:, used_indices] /= np.linalg.norm(self._data[:, used_indices], axis=1, keepdims=True)

    def kron_to_points(self, data_structure: DataStructureType, number_of_points: int) -> np.ndarray:
        """
        Maps the internal 2D ARRAY or symmetric matrix to a kron product

        :param data_structure: Used datastructure
        :param number_of_points: The amount of points for which the kron product should be built
        :return:
        """
        data_array = self.get_array(data_structure)
        if self.get_data_type() == DataObjectType.ARRAY_2D:
            kron_data_array = np.zeros((data_array.shape[0], number_of_points * len(data_structure), 1))
            kron_mapping = np.ones(number_of_points)[:, None]
            for index, point in enumerate(data_array):
                kron_data_array[index] = np.kron(kron_mapping, point[:, None])
            return kron_data_array
        elif self.get_data_type() == DataObjectType.SYM_CONV_MATRIX:
            kron_data_array = np.zeros(
                (data_array.shape[0], number_of_points * len(data_structure), number_of_points * len(data_structure))
            )
            kron_mapping = np.eye(number_of_points)

            for index, matrix in enumerate(data_array):
                kron_data_array[index] = np.kron(kron_mapping, matrix)

            return kron_data_array
        else:
            raise NotImplementedError

    def __str__(self):
        if self.get_data_type() == DataObjectType.ARRAY_2D and self.get_shape()[0] == 1:
            data_info: list[str] = []
            data_ele = self.get_array(self.get_data_structure())[0]
            for input_key in POSITION:
                if input_key in self.get_data_structure():
                    data_info.append(f"{input_key.value}: {data_ele[self._get_index(input_key)]}")

            rot_data = self.get_rotation_array(RotationOutputType.QUATERNION_W_LAST)[0]
            for key, ele in zip(ORIENTATION_QUAT_W_LAST, rot_data):
                data_info.append(f"{key.value}: {ele}")

            return f"DataObject({', '.join(data_info)})"
        else:
            return f"DataObject(shape={self.get_shape()})"

    def __repr__(self):
        return self.__str__()
