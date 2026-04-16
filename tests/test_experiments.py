def test_tp_kmp_initialization():
    """Test that TP-KMP can be initialized with real data without errors"""
    from interactive_incremental_learning import ConfigParams, initialize_tpkmp

    ################################################### Initialize Model
    params = ConfigParams()
    tp_kmp_model, tp_kmp_params, frames = initialize_tpkmp(params=params)

    # Basic sanity checks
    assert tp_kmp_model is not None
    assert tp_kmp_params is not None
    assert frames is not None

    # Check that the model has the expected structure
    assert hasattr(tp_kmp_model, "data_handler")
    assert hasattr(tp_kmp_model, "tp_kmp")  # Correct attribute name for KMP list
    assert len(tp_kmp_model.tp_kmp) > 0  # Should have KMPs for each frame


class TestExperimentExecution:
    """Test that all experiments run successfully without plotting"""

    def test_experiment_0_generalization(self, setup_model):
        """Test experiment 0 (generalization) runs without errors"""
        from interactive_incremental_learning.experiments.generalization import GeneralizationExperiment

        tp_kmp_model, tp_kmp_params, frames, config = setup_model

        experiment = GeneralizationExperiment()
        result = experiment.run(
            tp_kmp_model=tp_kmp_model,
            tp_kmp_params=tp_kmp_params,
            frames_used_for_demonstrations=frames,
            params=config,
            plot=False,
        )

        assert result is not None

    def test_experiment_1_adding_via_points(self, setup_model):
        """Test experiment 1 (adding via-points) runs without errors"""
        from interactive_incremental_learning.experiments.adding_via_points import AddViaPointsExperiment

        tp_kmp_model, tp_kmp_params, frames, config = setup_model

        experiment = AddViaPointsExperiment()
        result = experiment.run(
            tp_kmp_model=tp_kmp_model,
            tp_kmp_params=tp_kmp_params,
            frames_used_for_demonstrations=frames,
            params=config,
            plot=False,
        )

        assert result is not None

    def test_experiment_2_adding_frames(self, setup_model):
        """Test experiment 2 (adding frames) runs without errors"""
        from interactive_incremental_learning.experiments.adding_frames import AddFramesExperiment

        tp_kmp_model, tp_kmp_params, frames, config = setup_model

        experiment = AddFramesExperiment()
        result = experiment.run(
            tp_kmp_model=tp_kmp_model,
            tp_kmp_params=tp_kmp_params,
            frames_used_for_demonstrations=frames,
            params=config,
            plot=False,
        )

        assert result is not None

    def test_experiment_3_variable_stiffness(self, setup_model):
        """Test experiment 3 (variable stiffness) runs without errors"""
        from interactive_incremental_learning.experiments.calculate_variable_stiffness import (
            CalculateVariableStiffnessExperiment,
        )

        tp_kmp_model, tp_kmp_params, frames, config = setup_model

        experiment = CalculateVariableStiffnessExperiment()
        # This experiment doesn't return anything, just runs
        experiment.run(
            tp_kmp_model=tp_kmp_model,
            tp_kmp_params=tp_kmp_params,
            frames_used_for_demonstrations=frames,
            params=config,
            plot=False,
        )

        # If we get here without exceptions, the test passes
        assert True
