import numpy as np


class TestGMM:
    def test_gaussian_conditioning_basic(self):
        from interactive_incremental_learning.common.gmm import GaussianMixtureModel

        # Create simple 2D data
        np.random.seed(42)
        data = np.random.randn(100, 2)

        gmm = GaussianMixtureModel(n_components=2, random_state=42)
        gmm.fit(data)

        # Test conditioning
        x_in = np.array([[0.0]])
        mu_cond, cov_cond = gmm.gaussian_conditioning(0, x_in, d_in=[0], d_out=[1])

        assert mu_cond.shape[1] == 1  # Output dimension
        assert cov_cond.shape == (1, 1)  # Output covariance

    def test_gaussian_conditioning_shapes(self):
        from interactive_incremental_learning.common.gmm import GaussianMixtureModel

        np.random.seed(42)
        data = np.random.randn(50, 4)

        gmm = GaussianMixtureModel(n_components=1, random_state=42)
        gmm.fit(data)

        x_in = np.array([[0.0, 0.0]])
        mu_cond, cov_cond = gmm.gaussian_conditioning(0, x_in, d_in=[0, 1], d_out=[2, 3])

        assert mu_cond.shape == (1, 2)  # One sample, two output dims
        assert cov_cond.shape == (2, 2)  # 2x2 covariance matrix
