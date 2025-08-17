"""Tests for experiment module."""

import pytest
from unittest.mock import Mock, patch
from folie_a_deux.experiment import ExperimentResults, folie_a_deux, run_ablation_study
from folie_a_deux.config import ExperimentConfig


class TestExperimentResults:
    """Test the ExperimentResults class."""

    def test_experiment_results_init(self):
        """Test ExperimentResults initialization."""
        results = ExperimentResults()

        assert results.rounds == []
        assert results.verifier_a is None
        assert results.verifier_b is None

    def test_add_round(self):
        """Test adding round results."""
        results = ExperimentResults()

        results.add_round(1, 0.8, 0.7, 0.9, 0.85)

        assert len(results.rounds) == 1
        round_data = results.rounds[0]
        assert round_data["round"] == 1
        assert round_data["accuracy_a"] == 0.8
        assert round_data["accuracy_b"] == 0.7
        assert round_data["agreement_dev"] == 0.9
        assert round_data["agreement_train"] == 0.85

    def test_get_final_accuracies_empty(self):
        """Test get_final_accuracies with no rounds."""
        results = ExperimentResults()
        acc_a, acc_b = results.get_final_accuracies()

        assert acc_a == 0.0
        assert acc_b == 0.0

    def test_get_final_accuracies_with_rounds(self):
        """Test get_final_accuracies with rounds."""
        results = ExperimentResults()
        results.add_round(1, 0.8, 0.7, 0.9, 0.85)
        results.add_round(2, 0.85, 0.75, 0.92, 0.88)

        acc_a, acc_b = results.get_final_accuracies()
        assert acc_a == 0.85
        assert acc_b == 0.75

    def test_get_final_agreement_empty(self):
        """Test get_final_agreement with no rounds."""
        results = ExperimentResults()
        agreement = results.get_final_agreement()

        assert agreement == 0.0

    def test_get_final_agreement_with_rounds(self):
        """Test get_final_agreement with rounds."""
        results = ExperimentResults()
        results.add_round(1, 0.8, 0.7, 0.9, 0.85)
        results.add_round(2, 0.85, 0.75, 0.92, 0.88)

        agreement = results.get_final_agreement()
        assert agreement == 0.92

    def test_summary_empty(self):
        """Test summary with no rounds."""
        results = ExperimentResults()
        summary = results.summary()

        assert summary["total_rounds"] == 0

    def test_summary_with_rounds(self):
        """Test summary with rounds."""
        results = ExperimentResults()
        results.add_round(1, 0.8, 0.7, 0.9, 0.85)
        results.add_round(2, 0.85, 0.75, 0.92, 0.88)

        summary = results.summary()

        assert summary["total_rounds"] == 2
        assert summary["final_accuracy_a"] == 0.85
        assert summary["final_accuracy_b"] == 0.75
        assert summary["final_agreement"] == 0.92
        assert summary["max_accuracy_a"] == 0.85
        assert summary["max_accuracy_b"] == 0.75
        assert summary["max_agreement"] == 0.92


class TestFolieADeux:
    """Test the folie_a_deux function."""

    @patch("folie_a_deux.experiment.get_dev_labeled")
    @patch("folie_a_deux.experiment.get_train_unlabeled")
    @patch("folie_a_deux.experiment.validate_dataset")
    @patch("folie_a_deux.experiment.Verifier")
    @patch("folie_a_deux.experiment.MIPROv2")
    @patch("folie_a_deux.experiment.evaluate")
    @patch("folie_a_deux.experiment.agreement_rate")
    def test_folie_a_deux_basic_run(
        self,
        mock_agreement_rate,
        mock_evaluate,
        mock_miprov2,
        mock_verifier,
        mock_validate,
        mock_get_train,
        mock_get_dev,
    ):
        """Test basic folie_a_deux run."""
        # Mock data
        mock_dev_data = [Mock()]
        mock_train_data = [Mock() for _ in range(100)]
        mock_get_dev.return_value = mock_dev_data
        mock_get_train.return_value = mock_train_data

        # Mock verifiers
        mock_verifier_instance = Mock()
        mock_verifier.return_value = mock_verifier_instance

        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.compile.return_value = mock_verifier_instance
        mock_miprov2.return_value = mock_optimizer

        # Mock evaluation functions
        mock_evaluate.return_value = 0.8
        mock_agreement_rate.return_value = 0.9

        # Create config
        config = ExperimentConfig(rounds=2, alpha=0.0)

        # Run experiment
        results = folie_a_deux(config)

        # Verify results
        assert isinstance(results, ExperimentResults)
        assert len(results.rounds) == 2
        assert results.verifier_a is not None
        assert results.verifier_b is not None

        # Verify data loading was called
        mock_get_dev.assert_called_once()
        mock_get_train.assert_called_once()
        mock_validate.assert_called()

    @patch("folie_a_deux.experiment.get_dev_labeled")
    @patch("folie_a_deux.experiment.get_train_unlabeled")
    @patch("folie_a_deux.experiment.validate_dataset")
    def test_folie_a_deux_config_overrides(
        self, mock_validate, mock_get_train, mock_get_dev
    ):
        """Test folie_a_deux with parameter overrides."""
        # Mock data
        mock_get_dev.return_value = [Mock()]
        mock_get_train.return_value = [Mock() for _ in range(100)]

        with (
            patch("folie_a_deux.experiment.Verifier"),
            patch("folie_a_deux.experiment.MIPROv2"),
            patch("folie_a_deux.experiment.evaluate", return_value=0.8),
            patch("folie_a_deux.experiment.agreement_rate", return_value=0.9),
        ):
            # Test parameter overrides
            results = folie_a_deux(
                config=None,  # Should create default config
                rounds=3,
                use_cot=True,
                alpha_anchor=0.5,
            )

            assert isinstance(results, ExperimentResults)

    @patch("folie_a_deux.experiment.get_dev_labeled")
    def test_folie_a_deux_error_handling(self, mock_get_dev):
        """Test folie_a_deux error handling."""
        # Mock data loading to raise exception
        mock_get_dev.side_effect = Exception("Data loading failed")

        config = ExperimentConfig(rounds=1)

        # Should raise the exception
        with pytest.raises(Exception, match="Data loading failed"):
            folie_a_deux(config)


class TestRunAblationStudy:
    """Test the run_ablation_study function."""

    @patch("folie_a_deux.experiment.folie_a_deux")
    def test_run_ablation_study(self, mock_folie_a_deux):
        """Test ablation study execution."""
        # Mock folie_a_deux to return results
        mock_result = Mock()
        mock_folie_a_deux.return_value = mock_result

        config = ExperimentConfig()
        results = run_ablation_study(config)

        # Should test multiple alpha values
        expected_alphas = [0.0, 0.05, 0.1, 0.2, 0.5]
        assert len(results) == len(expected_alphas)

        # Check that folie_a_deux was called for each alpha
        assert mock_folie_a_deux.call_count == len(expected_alphas)

        # Check result keys
        for alpha in expected_alphas:
            key = f"alpha_{alpha}"
            assert key in results
            assert results[key] == mock_result

    @patch("folie_a_deux.experiment.folie_a_deux")
    def test_run_ablation_study_with_errors(self, mock_folie_a_deux):
        """Test ablation study with some experiments failing."""

        # Mock folie_a_deux to fail for some alpha values
        def side_effect(config):
            if config.alpha == 0.1:
                raise Exception("Test error")
            return Mock()

        mock_folie_a_deux.side_effect = side_effect

        config = ExperimentConfig()
        results = run_ablation_study(config)

        # Should have results for successful runs and None for failed ones
        assert results["alpha_0.1"] is None
        assert results["alpha_0.0"] is not None

    def test_run_ablation_study_default_config(self):
        """Test ablation study with default config."""
        with patch("folie_a_deux.experiment.folie_a_deux") as mock_folie_a_deux:
            mock_folie_a_deux.return_value = Mock()

            # Should create default config if None provided
            results = run_ablation_study(None)

            assert len(results) == 5  # Should test 5 alpha values
