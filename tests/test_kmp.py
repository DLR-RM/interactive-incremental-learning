import numpy as np
import pytest


class TestKMP:
    def test_kmp_initialization(self):
        from interactive_incremental_learning.common.kmp import Kmp

        kmp = Kmp()

        # Test default parameters
        assert kmp.gmm_n_components == 5
        assert kmp.N == 100
        assert kmp.l == 0.1
        assert kmp.h == 1.0
        assert kmp.lambda1 == 0.1
        assert kmp.lambda2 == 100
        assert kmp.alpha == 100
        assert kmp.nb_via == 0

    def test_kmp_custom_parameters(self):
        from interactive_incremental_learning.common.kmp import Kmp

        kmp = Kmp(gmm_n_components=3, N=50, length_scale=0.2, h=2.0, lambda1=0.05, lambda2=200, alpha=50)

        assert kmp.gmm_n_components == 3
        assert kmp.N == 50
        assert kmp.l == 0.2
        assert kmp.h == 2.0
        assert kmp.lambda1 == 0.05
        assert kmp.lambda2 == 200
        assert kmp.alpha == 50


class TestEdgeCases:
    def test_singular_covariance_handling(self):
        from interactive_incremental_learning.common.gmm import GaussianMixtureModel

        # Create data that might lead to singular covariance
        np.random.seed(42)
        data = np.zeros((10, 2))
        data[:, 0] = np.linspace(0, 1, 10)  # Perfectly linear

        gmm = GaussianMixtureModel(n_components=1, random_state=42, reg_covar=1e-6)
        gmm.fit(data)

        # Should not crash with singular matrix
        x_in = np.array([[0.5]])
        mu_cond, cov_cond = gmm.gaussian_conditioning(0, x_in, d_in=[0], d_out=[1])

        assert np.isfinite(mu_cond).all()
        assert np.isfinite(cov_cond).all()

    def test_non_normalized_quaternion_handling(self):
        from interactive_incremental_learning.common.tp_math import make_skew_matrix_for_w_first

        # Test with non-normalized quaternion
        quat = np.array([2.0, 0.0, 0.0, 0.0])  # Not normalized
        skew = make_skew_matrix_for_w_first(quat)

        # Should still work (function doesn't enforce normalization)
        assert skew.shape == (4, 4)
        assert np.isfinite(skew).all()

    def test_empty_data_handling(self):
        from interactive_incremental_learning.common.gmm import GaussianMixtureModel

        # Test with minimal data
        np.random.seed(42)
        data = np.random.randn(2, 2)  # Very small dataset

        with pytest.raises((ValueError, RuntimeError)):
            # Should fail with insufficient data
            gmm = GaussianMixtureModel(n_components=5, random_state=42)
            gmm.fit(data)

    def test_kernel_with_zero_length_scale(self):
        from interactive_incremental_learning.common.kernel import matern_kernel_p2

        x1 = np.array([[0.0]])
        x2 = np.array([[1.0]])

        # Very small length scale should still work
        K = matern_kernel_p2(x1, x2, length_scale=1e-10, h=1.0)

        assert np.isfinite(K).all()
        assert K.shape == (1, 1)
