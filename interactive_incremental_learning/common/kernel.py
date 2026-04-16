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
Script for kernel management

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

available_kernels = ["matern2"]


def matern_kernel_p2(x1: np.ndarray, x2: np.ndarray, length_scale: np.ndarray | float, h: float = 1.0) -> np.ndarray:
    """Implementation of the Matern kernel with p=2."""

    diff = np.repeat(x1[:, np.newaxis, :], x2.shape[0], axis=1) - np.repeat(x2[np.newaxis, :, :], x1.shape[0], axis=0)
    length_scale = np.array(length_scale)
    if len(length_scale.shape) == 0:
        length_scale = length_scale.reshape((1,))
    squared_dist = 5 * np.einsum("...i,ij,...j->...", diff, np.diag(1 / np.square(length_scale)), diff)
    dist = np.sqrt(squared_dist)
    return h**2 * (1 + dist + squared_dist / 3) * np.exp(-dist)


def kernel_matrix(
    x1: np.ndarray,
    x2: np.ndarray,
    length_scale: np.ndarray | float,
    h: float,
    kernel_function: str,
    kron: np.ndarray | None = None,
) -> np.ndarray:
    """Computation of the kernel matrix for two inputs."""
    if kernel_function == available_kernels[0]:
        K = matern_kernel_p2(x1, x2, length_scale, h)

    return np.kron(K, kron) if kron is not None else K
