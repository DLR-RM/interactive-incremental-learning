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
Script for math helper functions

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


def make_skew_matrix_for_w_last(reference_quat_w_last: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion in w last representation into a skew matrix w last presentation.

    :param reference_quat_w_last : quaternion in w last
    :return: A skew matrix which can be used on other quaternions with w last
    """

    if len(reference_quat_w_last.shape) != 1 or reference_quat_w_last.shape[0] != 4:
        raise ValueError(f"Only a 4-component quaternion is supported, got shape: {reference_quat_w_last.shape}")

    w = reference_quat_w_last[3]
    wx = reference_quat_w_last[0]
    wy = reference_quat_w_last[1]
    wz = reference_quat_w_last[2]
    # this is not the same matrix as for w_first!
    return np.array([[w, -wz, wy, wx], [wz, w, -wx, wy], [-wy, wx, w, wz], [-wx, -wy, -wz, w]])


def make_skew_matrix_for_w_first(reference_quat_w_first: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion in w first representation into a skew matrix w first presentation.

    :param reference_quat_w_first : quaternion in w first
    :return: A skew matrix which can be used on other quaternions with w first
    """

    if len(reference_quat_w_first.shape) != 1 or reference_quat_w_first.shape[0] != 4:
        raise ValueError(f"Only a 4-component quaternion is supported, got shape: {reference_quat_w_first.shape}")

    w = reference_quat_w_first[0]
    wx = reference_quat_w_first[1]
    wy = reference_quat_w_first[2]
    wz = reference_quat_w_first[3]
    # this is not the same matrix as for w_last!
    return np.array([[w, -wx, -wy, -wz], [wx, w, -wz, wy], [wy, wz, w, -wx], [wz, -wy, wx, w]])


def convert_w_last_to_w_first(quat_w_last: np.ndarray) -> np.ndarray:
    """
    Converts a quaternion from w last to w first

    :param quat_w_last: A quaternion in w last [wx, wy, wz, w]
    :return: A quaternion in w first [w, wx, wy, wz]
    """
    if len(quat_w_last.shape) != 1 or quat_w_last.shape[0] != 4:
        raise ValueError(f"Only a 4-component quaternion is supported, got shape: {quat_w_last.shape}")

    return quat_w_last[[3, 0, 1, 2]]


def convert_w_first_to_w_last(quat_w_first: np.ndarray) -> np.ndarray:
    """
    Converts a quaternion from w first to w last

    :param quat_w_first: A quaternion in w first [w, wx, wy, wz]
    :return: A quaternion in w last [wx, wy, wz, w]
    """
    if len(quat_w_first.shape) != 1 or quat_w_first.shape[0] != 4:
        raise ValueError(f"Only a 4-component quaternion is supported, got shape: {quat_w_first.shape}")

    return quat_w_first[[1, 2, 3, 0]]
