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
Script for Gaussian Mixture regression.

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
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


class GaussianMixtureModel(GaussianMixture):
    def __init__(self, **kwargs):
        """
        Inherit from from sklearn.mixture.GaussianMixture and add gaussian_mixture_regression
        https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
        """
        super().__init__(**kwargs)

    def gaussian_conditioning(
        self, index: int, x_in: np.ndarray, d_in: list | None = None, d_out: list | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        :param index: GMM component index
        :param x_in: input data
        :param d_in: x_in indices
        :param d_out: indexes to condition onto
        :return: tuple of (conditional mean, conditional covariance)
        """
        if d_in is None:
            d_in = list(range(0, 1))
        if d_out is None:
            d_out = list(range(1, 2))

        mean = self.means_[index]
        cov = self.covariances_[index]
        nb_dim_in = np.size(d_in)
        nb_dim_out = np.size(d_out)

        mu_ii = mean[d_in].reshape(nb_dim_in, -1)
        mu_oo = mean[d_out].reshape(nb_dim_out, -1)
        cov_ii = cov[np.ix_(d_in, d_in)]
        prec_ii = np.linalg.inv(cov_ii)
        cov_io = cov[np.ix_(d_in, d_out)]
        cov_oi = cov_io.T
        cov_oo = cov[np.ix_(d_out, d_out)]

        # conditional distribution
        mu_cond = mu_oo + cov_oi @ prec_ii @ (x_in.T - mu_ii)
        mu_cond = mu_cond.T  # features as columns
        cov_cond = cov_oo - cov_oi @ prec_ii @ cov_io

        return mu_cond, cov_cond

    def gaussian_mixture_regression(
        self,
        x_in: np.ndarray,
        d_in: list | None = None,
        d_out: list | None = None,
        N: int | None = None,
        single_gaussian: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | float]:
        """
        :param x_in: input data
        :param d_in: x_in indices
        :param d_out: indexes to condition onto
        :param N: len(x_in) or shorter ? Is that even required ?
        :param single_gaussian: if True use moment matching (one Gaussian), else return whole GMM
        """
        if d_in is None and d_out is None:
            d_in = [0]
            d_out = [1]
        else:
            if d_in is None or d_out is None:
                raise ValueError("Both d_in and d_out must be provided if either is specified")

        if len(x_in.shape) > 2:
            raise ValueError(f"The input must be one or two dimensional, got dimensionality of {len(x_in.shape)}")
        if len(x_in.shape) == 2:
            if x_in.shape[1] != len(d_in):
                raise ValueError(
                    f"The input must have the same size as the requested list of dimensions! "
                    f"But got input size of {x_in.shape[1]} and requested dimension of {len(d_in)}"
                )
        else:  # For 1D Input
            if x_in.ndim != len(d_in):
                raise ValueError(
                    f"The input must have the same size as the requested list of dimensions! "
                    f"But got input size of {x_in.ndim} and requested dimension of {len(d_in)}"
                )

        # why in var at all
        if N is None:
            if len(x_in.shape) == 2:
                N = np.shape(x_in)[0]
            else:
                N = x_in.shape[0]

        nb_dim_in = np.size(d_in)
        nb_dim_out = np.size(d_out)

        if self.means_.shape[1] != nb_dim_out + nb_dim_in:
            raise ValueError(
                f"All used input and output dimensions have to be used in the definition of what is in and output! "
                f"Here means shape is {self.means_.shape[1]} and sum of in- and output dimensions is {nb_dim_in + nb_dim_out}"
            )

        # initialize variables to store conditional distributions
        mu_cond = np.zeros((N, len(d_out), self.n_components))
        sigma_cond = np.zeros(
            (len(d_out), len(d_out), self.n_components)
        )  # doesn't need N because the covariance of the conditional is not input-dep.

        # to store moment matching approximation
        mu = np.zeros((N, len(d_out)))
        sigma = np.zeros((N, len(d_out), len(d_out)))

        h = np.zeros((N, self.n_components))

        for i in range(0, self.n_components):
            # marginal distribution of the input variable
            mu_ii = self.means_[i, np.ix_(d_in)].reshape(nb_dim_in, -1)
            cov_ii = self.covariances_[i][np.ix_(d_in, d_in)]

            # conditional distribution for each Gaussian
            mu_cond[:, :, i], sigma_cond[:, :, i] = self.gaussian_conditioning(i, x_in, d_in, d_out)

            # prior update
            if len(x_in.shape) > 1 and x_in.shape[1] == 1:
                h[:, i] = self.weights_[i] * multivariate_normal.pdf(x_in[:, 0], mean=mu_ii.flatten(), cov=cov_ii)
            else:
                h[:, i] = self.weights_[i] * multivariate_normal.pdf(x_in, mean=mu_ii.flatten(), cov=cov_ii)

        h = h / np.sum(h, axis=1)[:, None]  # priors must sum to 1

        if single_gaussian:
            # moment matching to approximate multiple gaussians with just one
            for i in range(N):
                mu[i, :] = mu_cond[i, :, :] @ h[i, :]
                sigma_tmp = np.zeros((nb_dim_out, nb_dim_out))
                for n in range(self.n_components):
                    sigma_tmp += h[i, n] * (sigma_cond[:, :, n] + np.outer(mu_cond[i, :, n], mu_cond[i, :, n]))
                sigma[i, :, :] = sigma_tmp - np.outer(mu[i, :], mu[i, :])
            return mu, sigma, 1.0
        else:
            # return multiple gaussians for new GMM
            # mu_cond, sigma_cond and h (normalized)
            return mu_cond, sigma_cond, h
