import numpy as np


class TestKernel:
    def test_matern_kernel_p2_identity(self):
        from interactive_incremental_learning.common.kernel import matern_kernel_p2

        x = np.array([[0.0], [1.0], [2.0]])
        length_scale = 1.0
        h = 1.0

        K = matern_kernel_p2(x, x, length_scale, h)

        # Kernel matrix should be symmetric
        np.testing.assert_array_almost_equal(K, K.T)

        # Diagonal elements should be h^2 = 1.0 for Matern kernel
        expected_diag = h**2
        np.testing.assert_array_almost_equal(np.diag(K), expected_diag)

    def test_matern_kernel_properties(self):
        from interactive_incremental_learning.common.kernel import matern_kernel_p2

        x1 = np.array([[0.0]])
        x2 = np.array([[1.0]])
        length_scale = 1.0
        h = 2.0

        K = matern_kernel_p2(x1, x2, length_scale, h)

        # Should be positive
        assert np.all(K > 0)

        # Value should depend on h parameter
        assert K.shape == (1, 1)

    def test_kernel_matrix_function(self):
        from interactive_incremental_learning.common.kernel import kernel_matrix

        x1 = np.array([[0.0], [1.0]])
        x2 = np.array([[0.5], [1.5]])
        length_scale = 1.0
        h = 1.0

        K = kernel_matrix(x1, x2, length_scale, h, "matern2")

        assert K.shape == (2, 2)
        assert np.all(K > 0)

    def test_kernel_matrix_with_kron(self):
        from interactive_incremental_learning.common.kernel import kernel_matrix

        x1 = np.array([[0.0]])
        x2 = np.array([[1.0]])
        length_scale = 1.0
        h = 1.0
        kron_matrix = np.eye(3)

        K = kernel_matrix(x1, x2, length_scale, h, "matern2", kron=kron_matrix)

        # Result should be Kronecker product
        assert K.shape == (3, 3)
