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
Script for Kernelized Movement Primities

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
import scipy.linalg as sp

from interactive_incremental_learning.common.gmm import GaussianMixtureModel
from interactive_incremental_learning.common.kernel import available_kernels, kernel_matrix


def _block_diagonal_inv(matrix, block_size):
    """Invert a block-diagonal matrix by inverting each block independently."""
    n = matrix.shape[0]
    result = np.zeros_like(matrix)
    for i in range(0, n, block_size):
        result[i : i + block_size, i : i + block_size] = np.linalg.inv(matrix[i : i + block_size, i : i + block_size])
    return result


class Kmp:
    def __init__(
        self,
        gmm_n_components: int = 5,
        N: int = 100,
        length_scale: float = 0.1,
        h: float = 1.0,
        lambda1: float = 0.1,
        lambda2: float = 100,
        alpha: float = 100,
        kernel_function: str = "matern2",
        epi_reg: float | None = None,
    ):
        """
        A class with basic functionalities of kernelized movement primitives.
        See https://arxiv.org/pdf/1708.08638.pdf

        :param gmm_n_components: Nb of gaussians in the GMM
        :param N: Number of sample points for Gaussian Mixture Regression
        :param length_scale: length scale of the kernel
        :param h: proportionnal kernel scaling factor
        :param lambda1: E(xi(s*)) = k* (K + lambda*Sigma)^-1 mu (21)
        :param lambda2: D(xi(s*)) = N/lambda (k(s*, s*) - k*(K + lambda*Sigma)^-1 k*^T) (26)
        :param alpha: KMP covariance prediction proportionnal scaling
        :param kernel_function: "matern2"
        :param epi_reg: Regularization sigma_epi^2 for epistemic uncertainty prediction
                        E_epi(xi(s*)) = k* (K + sigma_epi^2 I)^-1 mu. If None is passed,
                        lambda1 is used as default.
        """

        self.gmm_n_components = gmm_n_components
        self.N = N
        self.l: float | np.ndarray = length_scale
        self.h = h
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha
        self.nb_via = 0  # start with zero via-points

        if kernel_function not in available_kernels:
            raise ValueError(f"Kernel function '{kernel_function}' not recognized. Options: {available_kernels}")
        self.kernel_function = kernel_function

        # Set by default to lamda1
        if epi_reg is None:
            self.epi_reg = lambda1
        else:
            self.epi_reg = epi_reg

    def __repr__(self):
        return (
            f"KMP with:\n\tgmm_n_components = {self.gmm_n_components}"
            f"\n\tN = {self.N}\n\tl = {self.l}"
            f"\n\tlambda1 = {self.lambda1}\n\tlambda2 = {self.lambda2}"
            f"\n\talpha = {self.alpha}"
        )

    def fit(
        self,
        data: np.ndarray,
        d_in: list,
        d_out: list,
        gmm_init: tuple = (None, None, None),
        x_in: np.ndarray | None = None,
    ):
        """
        Initializes reference trajectory distribution of the KMP.
        (Using GMR as in the original paper, but only needs means and covariances so it can be done with other methods).

        :param data: List of n_feature-dimensional data points. array-like of shape (n_samples, n_features)
        :param d_in: List of input indices for regression
        :param d_out: List of output indices for regression
        :param gmm_init: Tuple of GMM initialization values
        :param x_in: data for regression
        """

        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be of type np.ndarray, but got {type(data)} instead!")

        means_init, weights_init, precisions_init = gmm_init

        # Train GMM on demonstration data
        gmm = GaussianMixtureModel(
            n_components=self.gmm_n_components,
            covariance_type="full",
            reg_covar=1e-5,
            init_params="kmeans",
            means_init=means_init,
            weights_init=weights_init,
            precisions_init=precisions_init,
        )
        gmm.fit(data)

        # GMR
        self.nb_dim_in = np.size(d_in)
        self.nb_dim_out = np.size(d_out)

        if isinstance(self.l, (np.ndarray, list)):
            self.l = np.array(self.l)
        else:
            self.l = self.l * np.ones(self.nb_dim_in)

        if x_in is None:
            if self.nb_dim_in == 1:
                x_in = np.linspace(0, 1, self.N)
            else:
                x_in = np.zeros((self.N, self.nb_dim_in))
                for i in range(0, self.N):
                    state = np.random.choice(self.gmm_n_components, p=gmm.weights_)
                    x_in[i, :] = np.random.multivariate_normal(
                        gmm.means_[state, 0 : self.nb_dim_in], gmm.covariances_[state, 0 : self.nb_dim_in, 0 : self.nb_dim_in]
                    )

        if not isinstance(x_in, np.ndarray):
            raise TypeError(f"x_in must be of type np.ndarray, but got {type(x_in)} instead!")
        if x_in.shape[0] != self.N:
            raise ValueError(
                f"The first dimension of the input must match the number of sample points! "
                f"But got {x_in.shape[0]} as first dimension of the input and {self.N} number of sample points."
            )
        if len(x_in.shape) >= 2 and x_in.shape[1] != self.nb_dim_in:
            raise ValueError(
                f"The second dimension of the input must match the given input size! "
                f"But got {x_in.shape[1]} as second dimension of the input and {self.nb_dim_in} as given input size."
            )
        mu, sigma, _ = gmm.gaussian_mixture_regression(x_in, d_in, d_out, self.N)

        # Block-ize GMR output
        mu_block = mu.reshape(-1, 1)  # mean
        pntr_sigma = []
        for i in range(0, sigma.shape[0]):  # covariance
            pntr_sigma.append(sigma[i, :, :])
        sigma_block = sp.block_diag(*pntr_sigma)

        # need a 2d array because input might be multi-dim
        self.x_in = x_in.reshape(-1, self.nb_dim_in)
        # this is kept for analysis, could be discarded otherwise
        self.model = gmm
        self.mu = mu
        self.sigma = sigma
        self.mu_block = mu_block
        self.sigma_block = sigma_block
        self.update_K()

    def update_K(self):
        """Computes kernel matrix for all outputs."""
        I_O = np.eye(self.nb_dim_out)
        K_reduced = kernel_matrix(self.x_in, self.x_in, self.l, self.h, self.kernel_function)
        self.K = np.kron(K_reduced, I_O)

        self.invK = np.linalg.inv(self.K + self.lambda1 * self.sigma_block)
        # Used for the covariance prediction, difference is lambda2; might be relaxed if we assume lambda2=lambda1
        self.invK2 = np.linalg.inv(self.K + self.lambda2 * self.sigma_block)
        # term used for the computation of epistemic uncertainty
        self.invK_epi = np.linalg.inv(self.K + self.epi_reg * np.eye((self.N + self.nb_via) * self.nb_dim_out))
        self.invK_epi_reduced = np.linalg.inv(K_reduced + self.epi_reg * np.eye(self.N + self.nb_via))

    def update_inputs(self, x_test: np.ndarray):
        """Computes K_s and K_ss, the kernel matrices that depend on test inputs."""

        self.x_test = x_test.reshape(-1, self.nb_dim_in)
        self.K_s_reduced = kernel_matrix(self.x_test, self.x_in, self.l, self.h, self.kernel_function)
        self.K_ss_reduced = kernel_matrix(self.x_test, self.x_test, self.l, self.h, self.kernel_function)

        I_O = np.eye(self.nb_dim_out)
        self.K_s = np.kron(self.K_s_reduced, I_O)
        self.K_ss = np.kron(self.K_ss_reduced, I_O)

    def mean(self) -> np.ndarray:
        """KMP mean prediction.
        return value is of dimension N x O"""

        self.mu_out = (self.K_s @ self.invK @ self.mu_block).reshape((-1, self.nb_dim_out))
        return self.mu_out

    def epistemic(self) -> np.ndarray:
        """KMP epistemic prediction.
        return value is of dimension ON x ON"""

        self.epi_uncertainty = self.alpha * (self.K_ss - self.K_s @ self.invK_epi @ self.K_s.T)
        return self.epi_uncertainty

    def aleatoric(self) -> np.ndarray:
        """KMP aleatoric prediction.
        return value is of dimension ON x ON"""

        epsilon = 1e-7 * np.eye((self.N + self.nb_via) * self.nb_dim_out)  # singular matrix preventing threshold

        self.al_uncertainty = self.alpha * (
            self.K_s
            @ np.linalg.inv(self.K @ np.linalg.inv(self.lambda2 * self.sigma_block + epsilon) @ self.K + self.K + epsilon)
            @ self.K_s.T
        )
        return self.al_uncertainty

    def cov(self) -> np.ndarray:
        """KMP covariance prediction.
        return value is of dimension ON x ON"""

        self.sigma_out = self.alpha * (self.K_ss - self.K_s @ self.invK2 @ self.K_s.T)
        return self.sigma_out

    def cov_diag(self) -> np.ndarray:
        """KMP covariance prediction, diagonal only.
        return value is of dimension (N_test * dim_out,)"""

        temp = self.K_s @ self.invK2
        diag = np.diag(self.K_ss) - np.einsum("ij,ij->i", temp, self.K_s)
        return self.alpha * diag

    def epistemic_diag(self) -> np.ndarray:
        """KMP epistemic prediction, diagonal only.
        return value is of dimension (N_test * dim_out,)"""

        temp = self.K_s @ self.invK_epi
        diag = np.diag(self.K_ss) - np.einsum("ij,ij->i", temp, self.K_s)
        return self.alpha * diag

    def aleatoric_diag(self) -> np.ndarray:
        """KMP aleatoric prediction, diagonal only.
        return value is of dimension (N_test * dim_out,)"""

        n = (self.N + self.nb_via) * self.nb_dim_out
        epsilon = 1e-7 * np.eye(n)

        inv_sigma = _block_diagonal_inv(self.lambda2 * self.sigma_block + epsilon, self.nb_dim_out)
        inner = self.K @ inv_sigma @ self.K + self.K + epsilon
        temp = np.linalg.solve(inner, self.K_s.T)
        diag = np.einsum("ij,ji->i", self.K_s, temp)
        return self.alpha * diag

    def predict(self, x_test):
        """Computes mean and covariance predictions for test input."""
        self.update_inputs(x_test)
        return self.mean(), self.cov()

    def add_viapoints(
        self,
        input_via: float | list | np.ndarray,
        output_via: float | list | np.ndarray,
        gamma: float = 1e-8,
    ) -> None:
        """Adds via-points to the KMP.
        :param input_via: Position where the via-point will be added (e. g. time) this can either be a list or only one value
        :param output_via: The via-point (e.g. in x,y,z), this can also be a list or only one value
        :param gamma: covariance value of the via-point, default is 1e-8
        """
        if not isinstance(input_via, np.ndarray):
            if isinstance(input_via, float):
                input_via = np.array([input_via])
            elif isinstance(input_via, list):
                pass
            else:
                raise TypeError(f"only np.ndarray or float values are allowed, not {type(input_via)}")
        if not isinstance(output_via, np.ndarray):
            if isinstance(output_via, float):
                output_via = np.array([output_via])
            elif isinstance(output_via, list):
                pass
            else:
                raise TypeError(f"only np.ndarray or float values are allowed, not {type(output_via)}")
        if len(input_via) != len(output_via):
            raise ValueError(f"input_via and output_via must have the same length, got {len(input_via)} and {len(output_via)}")

        I_O = np.eye(self.nb_dim_out, self.nb_dim_out)
        precision = float(gamma) * I_O

        ### Recompute K, k*, k**
        for i in range(len(input_via)):
            via_point = [input_via[i], output_via[i], float(gamma) * I_O]

            self.x_in = np.append(self.x_in, via_point[0])  # step 1)
            self.mu_block = np.append(self.mu_block, via_point[1].T)  # step 2)
            self.sigma = np.append(self.sigma, precision.reshape(1, self.nb_dim_out, -1), axis=0)  # step 3)
            self.nb_via += 1

        self.x_in = self.x_in.reshape(-1, self.nb_dim_in)

        # Blockize GMR output + the new covariance
        pntr_sigma = []
        for i in range(0, self.sigma.shape[0]):
            pntr_sigma.append(self.sigma[i, :, :])
        self.sigma_block = sp.block_diag(*pntr_sigma)
        self.update_K()
        self.update_inputs(self.x_test)
