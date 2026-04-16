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
Class for managing demonstration data

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

import numpy as np

from interactive_incremental_learning.common.data_object import DataObject, DataObjectType, DataStructureType


class DataHandler:
    def __init__(
        self,
        data: np.ndarray,
        data_structure: DataStructureType,
        data_type: DataObjectType,
        list_of_frames: DataObject | None = None,
    ):
        """
        :param data: Demonstration data for example given in 2D-Array format with Demos x variables
        :param data_structure: The order of the given variables, e.g. x,y,z
        :param data_type: Format of "data", for example DataObjectType.2D_ARRAY
        :param list_of_frames: List of coordinations of specific points of interest (frames) in order to tranform
        the demonstration onto those given points
        """
        self._data = data
        self._data_structure = data_structure
        self._data_type = data_type
        self.list_of_frames = list_of_frames
        if self.list_of_frames is not None and self.list_of_frames.get_data_type() != DataObjectType.ARRAY_2D:
            raise ValueError("The frame list must be a ARRAY 2D DataObject")
        self._data_object: DataObject | None = None

    def prepare_for_ml(self):
        """
        Prepares data for ML.
        Sets the number of demos for PlotUtility and reshapes the data as expected from the models.
        """
        if not isinstance(self._data, np.ndarray):
            self._data = np.array(self._data)
        if self._data.ndim == 3:
            D = self._data[0].shape[1]  # number of variables
        else:
            D = self._data.shape[1]  # number of variables

        # Concatenate and save as DataObject
        self._data_object = DataObject(
            data=self._data.reshape(-1, D),
            data_structure=self._data_structure,
            data_type=self._data_type,
        )

    def get_data(
        self,
        requested_data_structure: DataStructureType | None = None,
        reference_frame: DataObject | None = None,
    ) -> DataObject:
        """
        Returns a given data structure
        :param requested_data_structure: List of data you want to have. For example for time and pose ["t", "x", "y", "z"]
        :param reference_frame: Transforms the x, y, z data into the given reference frame, quaternions are not
                                handled, first value is a 3x3 rotation, second value is the translational vector
                                3 elements. (Rigid-body transformation).
        :return: DataObject with requested format
        """
        if self._data_object is None:
            raise RuntimeError("The data object cannot be None when calling this function! Call prepare_for_ml() first.")

        if reference_frame is not None:
            if self._data_object.get_data_type() != DataObjectType.ARRAY_2D:
                raise RuntimeError("The data object must be a ARRAY 2D!")
            new_data_object = self._data_object.map_to_different_data_object(reference_frame=reference_frame)
            assert requested_data_structure is not None
            if new_data_object.get_data_structure() != self.get_data_structure():
                return DataObject(
                    new_data_object.get_array(requested_data_structure), requested_data_structure, DataObjectType.ARRAY_2D
                )
            return new_data_object
        if requested_data_structure is not None and requested_data_structure != self.get_data_structure():
            return DataObject(
                self._data_object.get_array(requested_data_structure), requested_data_structure, DataObjectType.ARRAY_2D
            )
        return self._data_object

    def get_transformed_demonstration_for_frames_index(self, index: int) -> DataObject:
        """Return the demonstration data transformed into the coordinate frame at the given index."""
        if self.list_of_frames is None:
            raise ValueError("List of frames cannot be None when calling get_transformed_demonstration_for_frames_index!")
        return self.get_data(
            requested_data_structure=self._data_structure, reference_frame=self.list_of_frames.get_rows([index])
        )

    def get_data_structure(self) -> DataStructureType:
        """Return the data structure definition (list of InputType) for this handler."""
        return self._data_structure

    def get_frames(self) -> DataObject | None:
        """Return the reference frames DataObject, or None if no frames were provided."""
        return self.list_of_frames
