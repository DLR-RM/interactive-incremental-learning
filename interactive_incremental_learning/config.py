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
Script for configs

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
from typing import ClassVar

from interactive_incremental_learning.common.data_object import (
    ORIENTATION_QUAT_W_FIRST,
    POSITION,
    InputType,
)


@dataclass(frozen=True)
class ConfigParams:
    """Frozen dataclass holding data structure definitions for robot demonstrations, frames, and model I/O."""

    # Parameters
    robot_data_structure: ClassVar[list[InputType]] = [
        InputType.T,
        *POSITION,
        *ORIENTATION_QUAT_W_FIRST,
    ]  # Dataformat how we get it from pysara recording (for demos and frames)

    robot_control_data_structure: ClassVar[list[InputType]] = [*POSITION, *ORIENTATION_QUAT_W_FIRST]

    frames_data_structure: ClassVar[list[InputType]] = [
        *POSITION,
        *ORIENTATION_QUAT_W_FIRST,
    ]  # Dataformat of the frames, as they are recorded
    via_data_structure: ClassVar[list[InputType]] = [*POSITION, *ORIENTATION_QUAT_W_FIRST]

    input_data_structure: ClassVar[list[InputType]] = [
        InputType.T,
        *POSITION,
        *ORIENTATION_QUAT_W_FIRST,
    ]  # Dataformat after preprocessing
    training_input_data_structure: ClassVar[list[InputType]] = [InputType.T]  # Dataformat for KMPs input
    training_output_data_structure: ClassVar[list[InputType]] = [*POSITION, *ORIENTATION_QUAT_W_FIRST]
