"""Integration tests for the complete framework."""

import pytest
from unittest.mock import Mock, patch
import dspy
from folie_a_deux.config import ExperimentConfig
from folie_a_deux.data import get_dev_labeled, get_train_unlabeled
from folie_a_deux.metrics import (
    truth_accuracy,
    agreement_metric_factory,
    blended_metric_factory,
)
from folie_a_deux.evaluation import evaluate, agreement_rate
from folie_a_deux.experiment import folie_a_deux, ExperimentResults


class TestFrameworkIntegration:
    """Test integration between different framework components."""

    def test_config_data_integration(self):
        """Test that config and data components work together."""
        # Load config
        config = ExperimentConfig.from_env()
        config.validate()

        # Load data
        dev_set = get_dev_labeled()
        train_set = get_train_unlabeled()

        # Verify data matches config expectations
        assert len(dev_set) == 30  # As documented
        assert len(train_set) == 98  # 7 * 14 as documented

        # Verify data is properly formatted for experiment
        assert all(hasattr(ex, "claim") for ex in dev_set)
        assert all(hasattr(ex, "verdict") for ex in dev_set)
        assert all(hasattr(ex, "claim") for ex in train_set)
        assert all(not hasattr(ex, "verdict") for ex in train_set)

    def test_verifier_metrics_integration(self):
        """Test that verifiers work with metrics."""
        # Create mock verifiers
        verifier_a = Mock()
        verifier_b = Mock()

        # Mock predictions
        pred_a = Mock()
        pred_a.verdict = "yes"
        pred_b = Mock()
        pred_b.verdict = "no"

        verifier_a.return_value = pred_a
        verifier_b.return_value = pred_b

        # Test metrics work with mock verifiers
        example = dspy.Example(claim="test", verdict="yes").with_inputs("claim")

        # Test truth accuracy
        accuracy = truth_accuracy(example, pred_a)
        assert accuracy == 1.0

        # Test agreement metric
        agreement_metric = agreement_metric_factory(verifier_b)
        agreement_score = agreement_metric(example, pred_a)
        assert agreement_score == 0.0  # Different verdicts

        # Test blended metric
        blended_metric = blended_metric_factory(verifier_b, alpha=0.5)
        blended_score = blended_metric(example, pred_a)
        assert blended_score == 0.5  # 0.5 * agreement + 0.5 * truth

    def test_evaluation_integration(self):
        """Test that evaluation functions work with real data."""
        # Get real data
        dev_set = get_dev_labeled()[:5]  # Use subset for speed

        # Create mock verifier
        verifier = Mock()
        pred = Mock()
        pred.verdict = "yes"
        verifier.return_value = pred

        # Test evaluation
        score = evaluate(verifier, dev_set, display_progress=False)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

        # Test agreement rate
        verifier_b = Mock()
        pred_b = Mock()
        pred_b.verdict = "yes"
        verifier_b.return_value = pred_b

        agreement = agreement_rate(verifier, verifier_b, dev_set)
        assert isinstance(agreement, float)
        assert 0.0 <= agreement <= 1.0

    @patch("folie_a_deux.experiment.MIPROv2")
    @patch("folie_a_deux.experiment.evaluate")
    @patch("folie_a_deux.experiment.agreement_rate")
    def test_experiment_components_integration(
        self, mock_agreement_rate, mock_evaluate, mock_miprov2
    ):
        """Test that experiment integrates all components correctly."""
        # Mock the optimizer
        mock_optimizer = Mock()
        mock_verifier = Mock()
        mock_optimizer.compile.return_value = mock_verifier
        mock_miprov2.return_value = mock_optimizer

        # Mock evaluation functions
        mock_evaluate.return_value = 0.8
        mock_agreement_rate.return_value = 0.9

        # Create minimal config
        config = ExperimentConfig(rounds=1, alpha=0.0)

        # Run experiment
        result = folie_a_deux(config)

        # Verify result structure
        assert isinstance(result, ExperimentResults)
        assert len(result.rounds) == 1
        assert result.verifier_a is not None
        assert result.verifier_b is not None

        # Verify the round data
        round_data = result.rounds[0]
        assert round_data["round"] == 1
        assert "accuracy_a" in round_data
        assert "accuracy_b" in round_data
        assert "agreement_dev" in round_data
        assert "agreement_train" in round_data

    def test_config_environment_integration(self):
        """Test config loading from environment."""
        import os

        # Test with custom environment
        original_model = os.environ.get("MODEL")
        original_alpha = os.environ.get("ALPHA")

        try:
            os.environ["MODEL"] = "test_model"
            os.environ["ALPHA"] = "0.25"

            config = ExperimentConfig.from_env()

            assert config.model == "test_model"
            assert config.alpha == 0.25

        finally:
            # Restore original environment
            if original_model is not None:
                os.environ["MODEL"] = original_model
            elif "MODEL" in os.environ:
                del os.environ["MODEL"]

            if original_alpha is not None:
                os.environ["ALPHA"] = original_alpha
            elif "ALPHA" in os.environ:
                del os.environ["ALPHA"]

    def test_data_validation_integration(self):
        """Test that data validation catches real issues."""
        from folie_a_deux.data import validate_dataset

        # Test with real valid data
        dev_set = get_dev_labeled()
        train_set = get_train_unlabeled()

        # Should not raise
        validate_dataset(dev_set, require_labels=True)
        validate_dataset(train_set, require_labels=False)

        # Test with invalid data
        invalid_example = dspy.Example(not_a_claim="invalid").with_inputs("not_a_claim")

        with pytest.raises(ValueError):
            validate_dataset([invalid_example])

    def test_full_pipeline_mock_integration(self):
        """Test the full pipeline with mocked LLM calls."""
        # This tests that all components can be wired together
        # without actually calling an LLM

        with (
            patch("folie_a_deux.experiment.MIPROv2") as mock_miprov2,
            patch("folie_a_deux.experiment.evaluate") as mock_evaluate,
            patch("folie_a_deux.experiment.agreement_rate") as mock_agreement_rate,
        ):
            # Setup mocks
            mock_optimizer = Mock()
            mock_verifier = Mock()
            mock_optimizer.compile.return_value = mock_verifier
            mock_miprov2.return_value = mock_optimizer

            mock_evaluate.return_value = 0.75
            mock_agreement_rate.return_value = 0.85

            # Create config
            config = ExperimentConfig(
                model="test_model", rounds=2, alpha=0.1, use_cot=True
            )

            # Run experiment
            result = folie_a_deux(config)

            # Verify all components were called correctly
            assert len(result.rounds) == 2

            # Verify optimizer was called for both verifiers and both rounds
            # Should be called for: baseline A, baseline B, then 2 rounds * 2 verifiers = 6 total
            assert mock_miprov2.call_count == 6

            # Verify evaluation was called for each round
            assert mock_evaluate.call_count == 4  # 2 rounds * 2 verifiers

            # Verify agreement rate was called for each round
            assert (
                mock_agreement_rate.call_count == 4
            )  # 2 rounds * 2 agreement calculations


class TestErrorHandlingIntegration:
    """Test error handling across component boundaries."""

    def test_config_validation_errors(self):
        """Test that config validation catches integration issues."""
        # Test invalid alpha
        with pytest.raises(ValueError):
            config = ExperimentConfig(alpha=-0.1)
            config.validate()

        # Test invalid rounds
        with pytest.raises(ValueError):
            config = ExperimentConfig(rounds=0)
            config.validate()

    @patch("folie_a_deux.experiment.get_dev_labeled")
    def test_data_loading_error_propagation(self, mock_get_dev):
        """Test that data loading errors are properly handled."""
        # Mock data loading to fail
        mock_get_dev.side_effect = RuntimeError("Data loading failed")

        config = ExperimentConfig(rounds=1)

        with pytest.raises(RuntimeError, match="Data loading failed"):
            folie_a_deux(config)

    def test_metrics_error_handling_integration(self):
        """Test metrics error handling in realistic scenarios."""
        # Test with verifier that raises exceptions
        failing_verifier = Mock()
        failing_verifier.side_effect = Exception("Verifier failed")

        working_verifier = Mock()
        working_pred = Mock()
        working_pred.verdict = "yes"
        working_verifier.return_value = working_pred

        # Agreement metric should handle verifier failures gracefully
        agreement_metric = agreement_metric_factory(failing_verifier)
        example = dspy.Example(claim="test").with_inputs("claim")

        # Should return 0.0 on error, not raise exception
        score = agreement_metric(example, working_pred)
        assert score == 0.0


class TestPerformanceIntegration:
    """Test performance characteristics of integrated components."""

    def test_data_loading_performance(self):
        """Test that data loading is reasonably fast."""
        import time

        start_time = time.time()
        dev_set = get_dev_labeled()
        train_set = get_train_unlabeled()
        end_time = time.time()

        # Should load in under 1 second
        assert end_time - start_time < 1.0
        assert len(dev_set) == 30
        assert len(train_set) == 98

    def test_config_loading_performance(self):
        """Test that config loading is fast."""
        import time

        start_time = time.time()
        for _ in range(100):
            config = ExperimentConfig.from_env()
            config.validate()
        end_time = time.time()

        # 100 config loads should take under 0.1 seconds
        assert end_time - start_time < 0.1
