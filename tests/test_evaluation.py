"""Tests for evaluation module."""

from unittest.mock import Mock, patch
import dspy
from folie_a_deux.evaluation import (
    evaluate,
    agreement_rate,
    detailed_evaluation,
    compare_programs,
)


class TestEvaluate:
    """Test the evaluate function."""

    @patch("folie_a_deux.evaluation.dspy.Evaluate")
    def test_evaluate_success(self, mock_evaluator_class):
        """Test successful evaluation."""
        # Mock evaluator instance
        mock_evaluator = Mock()
        mock_evaluator.return_value = 0.85
        mock_evaluator_class.return_value = mock_evaluator

        # Mock program and dataset
        program = Mock()
        devset = [dspy.Example(claim="test", verdict="true")]

        # Test evaluation
        score = evaluate(program, devset)

        assert score == 0.85
        mock_evaluator_class.assert_called_once()
        mock_evaluator.assert_called_once_with(program)

    @patch("folie_a_deux.evaluation.dspy.Evaluate")
    def test_evaluate_error_handling(self, mock_evaluator_class):
        """Test evaluation error handling."""
        # Mock evaluator that raises exception
        mock_evaluator = Mock()
        mock_evaluator.side_effect = Exception("Test error")
        mock_evaluator_class.return_value = mock_evaluator

        # Mock program and dataset
        program = Mock()
        devset = [dspy.Example(claim="test", verdict="true")]

        # Test error handling
        score = evaluate(program, devset)

        assert score == 0.0


class TestAgreementRate:
    """Test the agreement_rate function."""

    def test_agreement_rate_empty_dataset(self):
        """Test agreement rate with empty dataset."""
        program_a = Mock()
        program_b = Mock()

        rate = agreement_rate(program_a, program_b, [])
        assert rate == 0.0

    def test_agreement_rate_full_agreement(self):
        """Test agreement rate with full agreement."""
        # Mock programs
        program_a = Mock()
        program_b = Mock()

        # Mock predictions
        pred_a = Mock()
        pred_a.verdict = "true"
        pred_b = Mock()
        pred_b.verdict = "TRUE"  # Different case, should still agree

        program_a.return_value = pred_a
        program_b.return_value = pred_b

        # Test dataset
        dataset = [
            dspy.Example(claim="claim1"),
            dspy.Example(claim="claim2"),
        ]

        rate = agreement_rate(program_a, program_b, dataset)
        assert rate == 1.0
        assert program_a.call_count == 2
        assert program_b.call_count == 2

    def test_agreement_rate_no_agreement(self):
        """Test agreement rate with no agreement."""
        # Mock programs
        program_a = Mock()
        program_b = Mock()

        # Mock predictions
        pred_a = Mock()
        pred_a.verdict = "true"
        pred_b = Mock()
        pred_b.verdict = "false"

        program_a.return_value = pred_a
        program_b.return_value = pred_b

        # Test dataset
        dataset = [dspy.Example(claim="claim1")]

        rate = agreement_rate(program_a, program_b, dataset)
        assert rate == 0.0

    def test_agreement_rate_partial_agreement(self):
        """Test agreement rate with partial agreement."""
        # Mock programs
        program_a = Mock()
        program_b = Mock()

        # Mock predictions that alternate agreement
        pred_a_true = Mock()
        pred_a_true.verdict = "true"
        pred_a_false = Mock()
        pred_a_false.verdict = "false"

        pred_b_true = Mock()
        pred_b_true.verdict = "true"
        pred_b_false = Mock()
        pred_b_false.verdict = "false"

        program_a.side_effect = [pred_a_true, pred_a_false]
        program_b.side_effect = [pred_b_true, pred_b_true]

        # Test dataset
        dataset = [
            dspy.Example(claim="claim1"),  # Both true - agree
            dspy.Example(claim="claim2"),  # A false, B true - disagree
        ]

        rate = agreement_rate(program_a, program_b, dataset)
        assert rate == 0.5

    def test_agreement_rate_error_handling(self):
        """Test agreement rate error handling."""
        # Mock programs where one raises exception
        program_a = Mock()
        program_a.side_effect = Exception("Test error")
        program_b = Mock()

        # Test dataset
        dataset = [dspy.Example(claim="claim1")]

        rate = agreement_rate(program_a, program_b, dataset)
        assert rate == 0.0  # Error counts as disagreement


class TestDetailedEvaluation:
    """Test the detailed_evaluation function."""

    def test_detailed_evaluation_success(self):
        """Test detailed evaluation with successful predictions."""
        # Mock program
        program = Mock()
        pred = Mock()
        pred.verdict = "true"
        program.return_value = pred

        # Test dataset
        devset = [
            dspy.Example(claim="claim1", verdict="true"),
            dspy.Example(claim="claim2", verdict="false"),
        ]

        results = detailed_evaluation(program, devset)

        assert results["total_examples"] == 2
        assert results["correct"] == 1
        assert results["incorrect"] == 1
        assert results["errors"] == 0
        assert results["accuracy"] == 0.5
        assert len(results["examples"]) == 2

    def test_detailed_evaluation_no_ground_truth(self):
        """Test detailed evaluation without ground truth."""
        # Mock program
        program = Mock()
        pred = Mock()
        pred.verdict = "true"
        program.return_value = pred

        # Test dataset without ground truth
        devset = [dspy.Example(claim="claim1")]

        results = detailed_evaluation(program, devset)

        assert results["total_examples"] == 1
        assert results["correct"] == 0
        assert results["incorrect"] == 0
        assert results["errors"] == 0
        assert results["accuracy"] == 0.0

    def test_detailed_evaluation_with_errors(self):
        """Test detailed evaluation with prediction errors."""
        # Mock program that raises exception
        program = Mock()
        program.side_effect = Exception("Test error")

        # Test dataset
        devset = [dspy.Example(claim="claim1", verdict="true")]

        results = detailed_evaluation(program, devset)

        assert results["total_examples"] == 1
        assert results["correct"] == 0
        assert results["incorrect"] == 0
        assert results["errors"] == 1
        assert results["accuracy"] == 0.0


class TestComparePrograms:
    """Test the compare_programs function."""

    def test_compare_programs_full_agreement(self):
        """Test program comparison with full agreement."""
        # Mock programs
        program_a = Mock()
        program_b = Mock()

        # Mock predictions
        pred_a = Mock()
        pred_a.verdict = "true"
        pred_b = Mock()
        pred_b.verdict = "true"

        program_a.return_value = pred_a
        program_b.return_value = pred_b

        # Test dataset
        devset = [
            dspy.Example(claim="claim1", verdict="true"),
            dspy.Example(claim="claim2", verdict="true"),
        ]

        results = compare_programs(program_a, program_b, devset)

        assert results["total_examples"] == 2
        assert results["agreement"] == 2
        assert results["disagreement"] == 0
        assert results["both_correct"] == 2
        assert results["both_incorrect"] == 0
        assert results["a_correct_b_incorrect"] == 0
        assert results["a_incorrect_b_correct"] == 0
        assert results["agreement_rate"] == 1.0

    def test_compare_programs_no_agreement(self):
        """Test program comparison with no agreement."""
        # Mock programs
        program_a = Mock()
        program_b = Mock()

        # Mock predictions
        pred_a = Mock()
        pred_a.verdict = "true"
        pred_b = Mock()
        pred_b.verdict = "false"

        program_a.return_value = pred_a
        program_b.return_value = pred_b

        # Test dataset
        devset = [dspy.Example(claim="claim1", verdict="true")]

        results = compare_programs(program_a, program_b, devset)

        assert results["agreement"] == 0
        assert results["disagreement"] == 1
        assert results["a_correct_b_incorrect"] == 1
        assert results["agreement_rate"] == 0.0

    def test_compare_programs_no_ground_truth(self):
        """Test program comparison without ground truth."""
        # Mock programs
        program_a = Mock()
        program_b = Mock()

        # Mock predictions
        pred_a = Mock()
        pred_a.verdict = "true"
        pred_b = Mock()
        pred_b.verdict = "true"

        program_a.return_value = pred_a
        program_b.return_value = pred_b

        # Test dataset without ground truth
        devset = [dspy.Example(claim="claim1")]

        results = compare_programs(program_a, program_b, devset)

        assert results["agreement"] == 1
        assert results["both_correct"] == 0  # No ground truth to compare
        assert results["agreement_rate"] == 1.0
