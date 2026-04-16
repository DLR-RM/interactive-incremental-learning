import numpy as np
import pytest


def create_mock_demonstration_data(n_demos=2, n_timesteps=100):
    """Create realistic mock data matching the pickle file structure"""
    demonstrations = np.zeros((n_demos, n_timesteps, 8), dtype=np.float32)

    # Time column (normalized)
    demonstrations[:, :, 0] = np.linspace(0, 1, n_timesteps)

    # Position columns (1-3)
    demonstrations[:, :, 1] = np.random.uniform(0.3, 0.9, (n_demos, n_timesteps))
    demonstrations[:, :, 2] = np.random.uniform(-0.6, 0.6, (n_demos, n_timesteps))
    demonstrations[:, :, 3] = np.random.uniform(0.15, 0.4, (n_demos, n_timesteps))

    # Quaternion columns (4-7) - generate valid unit quaternions
    for demo_idx in range(n_demos):
        for t in range(n_timesteps):
            quat = np.random.randn(4)
            quat = quat / np.linalg.norm(quat)  # Normalize to unit quaternion
            demonstrations[demo_idx, t, 4:8] = quat

    return demonstrations


def create_mock_frames_data(n_demos=2, n_frames=2):
    """Create realistic mock frame data"""
    frames = np.zeros((n_demos, n_frames, 7), dtype=np.float32)

    # Position (0-2)
    frames[:, :, 0] = np.random.uniform(0.3, 0.65, (n_demos, n_frames))
    frames[:, :, 1] = np.random.uniform(-0.6, 0.54, (n_demos, n_frames))
    frames[:, :, 2] = np.random.uniform(0.15, 0.17, (n_demos, n_frames))

    # Quaternion (3-6)
    for demo_idx in range(n_demos):
        for frame_idx in range(n_frames):
            quat = np.random.randn(4)
            quat = quat / np.linalg.norm(quat)
            frames[demo_idx, frame_idx, 3:7] = quat

    return frames


@pytest.fixture
def setup_model():
    """Initialize TP-KMP model for experiments with fast test parameters"""
    from interactive_incremental_learning import ConfigParams, initialize_tpkmp

    config = ConfigParams()

    # Use normal initialization
    tp_kmp_model, tp_kmp_params, frames = initialize_tpkmp(config)

    # Modify the number of test points for faster testing
    tp_kmp_params.number_of_test_points = 50  # Reduced from 500 to 50

    return tp_kmp_model, tp_kmp_params, frames, config
