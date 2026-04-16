import pickle

import numpy as np

from tests.conftest import create_mock_demonstration_data, create_mock_frames_data


class TestDataObject:
    def test_input_type_enum(self):
        from interactive_incremental_learning.common.data_object import InputType

        assert InputType.X.value == "x"
        assert InputType.T.value == "t"
        assert InputType.W.value == "w"  # Use .value instead of str()

    def test_data_structure_constants(self):
        from interactive_incremental_learning.common.data_object import (
            ORIENTATION_QUAT_W_FIRST,
            ORIENTATION_QUAT_W_LAST,
            POSITION,
            InputType,
        )

        assert len(POSITION) == 3
        assert POSITION == [InputType.X, InputType.Y, InputType.Z]

        assert len(ORIENTATION_QUAT_W_FIRST) == 4
        assert ORIENTATION_QUAT_W_FIRST[0] == InputType.W

        assert len(ORIENTATION_QUAT_W_LAST) == 4
        assert ORIENTATION_QUAT_W_LAST[-1] == InputType.W

    def test_rotation_output_type_conversion(self):
        from interactive_incremental_learning.common.data_object import ORIENTATION_QUAT_W_FIRST, RotationOutputType

        rot_type = RotationOutputType.QUATERNION_W_FIRST
        input_types = rot_type.to_input_types()

        assert input_types == ORIENTATION_QUAT_W_FIRST


class TestDataset:
    def test_dataset_exists(self):
        import os

        dataset_path = "data/tpkmp_all_data_storage.pickle"
        assert os.path.exists(dataset_path), f"Dataset not found at {dataset_path}"

    def test_dataset_structure(self):
        with open("data/tpkmp_all_data_storage.pickle", "rb") as f:
            data = pickle.load(f)

        assert isinstance(data, dict)
        assert "demonstrations" in data
        assert "frames" in data

        demos = data["demonstrations"]
        frames = data["frames"]

        # Test shapes match expected format
        assert len(demos.shape) == 3  # (n_demos, timesteps, features)
        assert demos.shape[2] == 8  # 8 features (time + 3pos + 4quat)

        assert len(frames.shape) == 3  # (n_demos, n_frames, frame_features)
        assert frames.shape[2] == 7  # 7 features (3pos + 4quat)

        # Test data quality
        assert not np.any(np.isnan(demos))
        assert not np.any(np.isnan(frames))
        assert np.all(np.isfinite(demos))
        assert np.all(np.isfinite(frames))

        # Test time normalization
        time_cols = demos[:, :, 0]
        for demo_idx in range(demos.shape[0]):
            time_col = time_cols[demo_idx]
            assert np.isclose(time_col[0], 0.0, atol=1e-6)
            assert np.isclose(time_col[-1], 1.0, atol=1e-6)
            # Time should be monotonically increasing (allowing for small floating point errors)
            time_diffs = np.diff(time_col)
            assert np.all(time_diffs >= -1e-8)  # Allow for tiny numerical errors


class TestConfig:
    def test_config_params_initialization(self):
        from interactive_incremental_learning.common.data_object import ORIENTATION_QUAT_W_FIRST, POSITION, InputType
        from interactive_incremental_learning.config import ConfigParams

        config = ConfigParams()

        # Test robot data structure
        expected_robot_structure = [InputType.T, *POSITION, *ORIENTATION_QUAT_W_FIRST]
        assert config.robot_data_structure == expected_robot_structure

        # Test frame data structure
        expected_frame_structure = [*POSITION, *ORIENTATION_QUAT_W_FIRST]
        assert config.frames_data_structure == expected_frame_structure

        # Test training structures
        assert config.training_input_data_structure == [InputType.T]
        assert config.training_output_data_structure == [*POSITION, *ORIENTATION_QUAT_W_FIRST]


class TestMockData:
    def test_mock_demonstration_data_shape(self):
        demos = create_mock_demonstration_data(n_demos=3, n_timesteps=50)

        assert demos.shape == (3, 50, 8)
        assert demos.dtype == np.float32

        # Check time column is normalized
        for demo_idx in range(3):
            time_col = demos[demo_idx, :, 0]
            assert np.isclose(time_col[0], 0.0)
            assert np.isclose(time_col[-1], 1.0)
            assert np.all(np.diff(time_col) > 0)  # Monotonic increasing

    def test_mock_frames_data_shape(self):
        frames = create_mock_frames_data(n_demos=2, n_frames=3)

        assert frames.shape == (2, 3, 7)
        assert frames.dtype == np.float32

    def test_mock_quaternion_normalization(self):
        demos = create_mock_demonstration_data(n_demos=1, n_timesteps=10)

        # Check quaternions are normalized
        for t in range(10):
            quat = demos[0, t, 4:8]
            norm = np.linalg.norm(quat)
            assert np.isclose(norm, 1.0, rtol=1e-6)

        frames = create_mock_frames_data(n_demos=1, n_frames=2)

        # Check frame quaternions are normalized
        for frame_idx in range(2):
            quat = frames[0, frame_idx, 3:7]
            norm = np.linalg.norm(quat)
            assert np.isclose(norm, 1.0, rtol=1e-6)
