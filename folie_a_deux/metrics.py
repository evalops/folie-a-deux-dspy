"""Metrics for evaluating verifier agreement and accuracy."""

from typing import Callable, Optional, Any
import logging
import dspy

logger = logging.getLogger(__name__)


def _normalize_verdict(verdict: Optional[str]) -> str:
    """Normalize a verdict string to lowercase and stripped."""
    return (verdict or "").strip().lower()


def truth_accuracy(
    example: dspy.Example, pred: dspy.Prediction, trace: Optional[Any] = None
) -> float:
    """
    Calculate accuracy against ground truth.

    Args:
        example: Ground truth example with 'verdict' field
        pred: Prediction with 'verdict' field
        trace: Optional trace (unused)

    Returns:
        1.0 if verdicts match, 0.0 otherwise
    """
    truth_verdict = _normalize_verdict(example.verdict)
    pred_verdict = _normalize_verdict(pred.verdict)

    accuracy = 1.0 if pred_verdict == truth_verdict else 0.0

    logger.debug(
        f"Truth accuracy: {accuracy} (truth={truth_verdict}, pred={pred_verdict})"
    )
    return accuracy


def agreement_metric_factory(other_program) -> Callable:
    """
    Create a metric that measures agreement with another program.

    Args:
        other_program: The other verifier to compare against

    Returns:
        Metric function that returns 1.0 for agreement, 0.0 for disagreement
    """

    def agreement_metric(
        example: dspy.Example, pred: dspy.Prediction, trace: Optional[Any] = None
    ) -> float:
        """Measure agreement between this prediction and another program's prediction."""
        try:
            other_pred = other_program(claim=example.claim)

            pred_verdict = _normalize_verdict(pred.verdict)
            other_verdict = _normalize_verdict(other_pred.verdict)

            agreement = 1.0 if pred_verdict == other_verdict else 0.0

            logger.debug(
                f"Agreement: {agreement} (pred={pred_verdict}, other={other_verdict})"
            )
            return agreement

        except Exception as e:
            logger.error(
                f"Error computing agreement for claim '{example.claim}': {e}",
                exc_info=True,
            )
            return 0.0

    return agreement_metric


def blended_metric_factory(other_program, alpha: float = 0.1) -> Callable:
    """
    Create a metric that blends agreement with truth accuracy.

    Args:
        other_program: The other verifier to compare against
        alpha: Weight for truth accuracy (1-alpha for agreement)

    Returns:
        Metric function that returns blended score
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Alpha must be between 0.0 and 1.0, got {alpha}")

    agreement_metric = agreement_metric_factory(other_program)

    def blended_metric(
        example: dspy.Example, pred: dspy.Prediction, trace: Optional[Any] = None
    ) -> float:
        """Compute blended metric combining agreement and truth accuracy."""
        try:
            agreement_score = agreement_metric(example, pred, trace)

            # Only use truth accuracy if example has ground truth
            if "verdict" in example:
                truth_score = truth_accuracy(example, pred, trace)
            else:
                truth_score = 0.0

            blended_score = (1 - alpha) * agreement_score + alpha * truth_score

            logger.debug(
                f"Blended metric: {blended_score} (agreement={agreement_score}, truth={truth_score}, alpha={alpha})"
            )
            return blended_score

        except Exception as e:
            logger.error(
                f"Error computing blended metric for claim '{example.claim}': {e}",
                exc_info=True,
            )
            return 0.0

    return blended_metric


def compute_confidence_interval(scores: list, confidence: float = 0.95) -> tuple:
    """
    Compute confidence interval for a list of scores.

    Args:
        scores: List of numeric scores
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    import statistics
    import math

    if not scores:
        return 0.0, 0.0, 0.0

    mean = statistics.mean(scores)

    if len(scores) == 1:
        return mean, mean, mean

    stdev = statistics.stdev(scores)
    n = len(scores)

    # Using t-distribution for small samples
    if n < 30:
        # Approximate t-value for common confidence levels
        t_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        t_value = t_values.get(confidence, 1.96)
    else:
        t_value = 1.96  # z-value for normal distribution

    margin_error = t_value * (stdev / math.sqrt(n))

    return mean, mean - margin_error, mean + margin_error
