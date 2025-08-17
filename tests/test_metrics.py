"""Tests for metrics module."""

import pytest
from unittest.mock import Mock
import dspy
from folie_a_deux.metrics import (
    _normalize_verdict,
    truth_accuracy,
    agreement_metric_factory,
    blended_metric_factory,
    compute_confidence_interval,
)


class TestNormalizeVerdict:
    """Test verdict normalization function."""

    def test_normalize_basic(self):
        """Test basic normalization."""
        assert _normalize_verdict("TRUE") == "true"
        assert _normalize_verdict("False") == "false"
        assert _normalize_verdict("  YES  ") == "yes"

    def test_normalize_none(self):
        """Test normalization with None input."""
        assert _normalize_verdict(None) == ""

    def test_normalize_empty(self):
        """Test normalization with empty input."""
        assert _normalize_verdict("") == ""
        assert _normalize_verdict("   ") == ""


class TestTruthAccuracy:
    """Test truth accuracy metric."""

    def test_truth_accuracy_correct(self):
        """Test accuracy with correct prediction."""
        example = dspy.Example(verdict="true")
        pred = Mock()
        pred.verdict = "TRUE"

        accuracy = truth_accuracy(example, pred)
        assert accuracy == 1.0

    def test_truth_accuracy_incorrect(self):
        """Test accuracy with incorrect prediction."""
        example = dspy.Example(verdict="true")
        pred = Mock()
        pred.verdict = "false"

        accuracy = truth_accuracy(example, pred)
        assert accuracy == 0.0

    def test_truth_accuracy_case_insensitive(self):
        """Test accuracy is case insensitive."""
        example = dspy.Example(verdict="True")
        pred = Mock()
        pred.verdict = "true"

        accuracy = truth_accuracy(example, pred)
        assert accuracy == 1.0


class TestAgreementMetricFactory:
    """Test agreement metric factory."""

    def test_agreement_metric_agree(self):
        """Test agreement metric when programs agree."""
        # Mock other program
        other_program = Mock()
        other_pred = Mock()
        other_pred.verdict = "true"
        other_program.return_value = other_pred

        # Create metric
        metric = agreement_metric_factory(other_program)

        # Test agreement
        example = dspy.Example(claim="test claim")
        pred = Mock()
        pred.verdict = "TRUE"

        agreement = metric(example, pred)
        assert agreement == 1.0
        other_program.assert_called_once_with(claim="test claim")

    def test_agreement_metric_disagree(self):
        """Test agreement metric when programs disagree."""
        # Mock other program
        other_program = Mock()
        other_pred = Mock()
        other_pred.verdict = "false"
        other_program.return_value = other_pred

        # Create metric
        metric = agreement_metric_factory(other_program)

        # Test disagreement
        example = dspy.Example(claim="test claim")
        pred = Mock()
        pred.verdict = "true"

        agreement = metric(example, pred)
        assert agreement == 0.0

    def test_agreement_metric_error_handling(self):
        """Test agreement metric error handling."""
        # Mock other program that raises exception
        other_program = Mock()
        other_program.side_effect = Exception("Test error")

        # Create metric
        metric = agreement_metric_factory(other_program)

        # Test error handling
        example = dspy.Example(claim="test claim")
        pred = Mock()
        pred.verdict = "true"

        agreement = metric(example, pred)
        assert agreement == 0.0


class TestBlendedMetricFactory:
    """Test blended metric factory."""

    def test_blended_metric_alpha_validation(self):
        """Test alpha validation in blended metric factory."""
        other_program = Mock()

        with pytest.raises(ValueError, match="Alpha must be between 0.0 and 1.0"):
            blended_metric_factory(other_program, alpha=-0.1)

        with pytest.raises(ValueError, match="Alpha must be between 0.0 and 1.0"):
            blended_metric_factory(other_program, alpha=1.5)

    def test_blended_metric_pure_agreement(self):
        """Test blended metric with alpha=0 (pure agreement)."""
        # Mock other program
        other_program = Mock()
        other_pred = Mock()
        other_pred.verdict = "true"
        other_program.return_value = other_pred

        # Create metric with alpha=0
        metric = blended_metric_factory(other_program, alpha=0.0)

        # Test with ground truth available
        example = dspy.Example(claim="test claim", verdict="false")
        pred = Mock()
        pred.verdict = "true"  # Agrees with other, disagrees with truth

        score = metric(example, pred)
        assert score == 1.0  # Pure agreement score

    def test_blended_metric_pure_truth(self):
        """Test blended metric with alpha=1 (pure truth)."""
        # Mock other program
        other_program = Mock()
        other_pred = Mock()
        other_pred.verdict = "false"
        other_program.return_value = other_pred

        # Create metric with alpha=1
        metric = blended_metric_factory(other_program, alpha=1.0)

        # Test with ground truth available
        example = dspy.Example(claim="test claim", verdict="true")
        pred = Mock()
        pred.verdict = "true"  # Disagrees with other, agrees with truth

        score = metric(example, pred)
        assert score == 1.0  # Pure truth score

    def test_blended_metric_mixed(self):
        """Test blended metric with mixed alpha."""
        # Mock other program
        other_program = Mock()
        other_pred = Mock()
        other_pred.verdict = "true"
        other_program.return_value = other_pred

        # Create metric with alpha=0.5
        metric = blended_metric_factory(other_program, alpha=0.5)

        # Test with ground truth available
        example = dspy.Example(claim="test claim", verdict="true")
        pred = Mock()
        pred.verdict = "true"  # Agrees with both other and truth

        score = metric(example, pred)
        assert score == 1.0  # Both agreement and truth are 1.0

    def test_blended_metric_no_truth(self):
        """Test blended metric without ground truth."""
        # Mock other program
        other_program = Mock()
        other_pred = Mock()
        other_pred.verdict = "true"
        other_program.return_value = other_pred

        # Create metric
        metric = blended_metric_factory(other_program, alpha=0.5)

        # Test without ground truth
        example = dspy.Example(claim="test claim")  # No verdict field
        pred = Mock()
        pred.verdict = "true"

        score = metric(example, pred)
        assert score == 0.5  # (1-0.5) * 1.0 + 0.5 * 0.0


class TestComputeConfidenceInterval:
    """Test confidence interval computation."""

    def test_confidence_interval_empty(self):
        """Test confidence interval with empty scores."""
        mean, lower, upper = compute_confidence_interval([])
        assert mean == 0.0
        assert lower == 0.0
        assert upper == 0.0

    def test_confidence_interval_single_score(self):
        """Test confidence interval with single score."""
        scores = [0.8]
        mean, lower, upper = compute_confidence_interval(scores)
        assert mean == 0.8
        assert lower == 0.8
        assert upper == 0.8

    def test_confidence_interval_multiple_scores(self):
        """Test confidence interval with multiple scores."""
        scores = [0.6, 0.7, 0.8, 0.9, 1.0]
        mean, lower, upper = compute_confidence_interval(scores)

        assert mean == 0.8
        assert lower < mean
        assert upper > mean
        assert lower >= 0.0
        assert upper <= 1.0

    def test_confidence_interval_custom_confidence(self):
        """Test confidence interval with custom confidence level."""
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        mean_95, lower_95, upper_95 = compute_confidence_interval(
            scores, confidence=0.95
        )
        mean_99, lower_99, upper_99 = compute_confidence_interval(
            scores, confidence=0.99
        )

        assert mean_95 == mean_99  # Same mean
        assert (upper_99 - lower_99) > (
            upper_95 - lower_95
        )  # Wider interval for higher confidence

    def test_confidence_interval_large_sample(self):
        """Test confidence interval with large sample (n >= 30)."""
        # Create a large sample to trigger the z-value branch
        scores = [0.5 + i * 0.01 for i in range(50)]  # 50 samples

        mean, lower, upper = compute_confidence_interval(scores, confidence=0.95)

        assert isinstance(mean, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower < mean < upper
        assert mean == sum(scores) / len(scores)

    def test_confidence_interval_unknown_confidence_level(self):
        """Test confidence interval with unknown confidence level uses default."""
        scores = [0.1, 0.2, 0.3]  # Small sample

        # Use an unknown confidence level
        mean, lower, upper = compute_confidence_interval(scores, confidence=0.85)

        # Should use default t-value of 1.96
        assert isinstance(mean, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower < mean < upper

    def test_confidence_interval_90_percent(self):
        """Test confidence interval with 90% confidence level."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]  # Small sample to use t-distribution

        mean_90, lower_90, upper_90 = compute_confidence_interval(
            scores, confidence=0.90
        )
        mean_95, lower_95, upper_95 = compute_confidence_interval(
            scores, confidence=0.95
        )

        # 90% CI should be narrower than 95% CI
        assert (upper_90 - lower_90) < (upper_95 - lower_95)
        assert mean_90 == mean_95

    def test_confidence_interval_99_percent(self):
        """Test confidence interval with 99% confidence level."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]  # Small sample to use t-distribution

        mean, lower, upper = compute_confidence_interval(scores, confidence=0.99)

        assert isinstance(mean, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower < mean < upper
