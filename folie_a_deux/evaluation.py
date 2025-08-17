"""Evaluation utilities for verifier performance."""

from typing import List, Dict, Any
import logging
import dspy
from .metrics import truth_accuracy, _normalize_verdict

logger = logging.getLogger(__name__)


def evaluate(
    program, devset: List[dspy.Example], metric=None, display_progress: bool = True
) -> float:
    """
    Evaluate a program on a development set.

    Args:
        program: The program to evaluate
        devset: List of labeled examples
        metric: Metric function (defaults to truth_accuracy)
        display_progress: Whether to show progress

    Returns:
        Average metric score
    """
    if metric is None:
        metric = truth_accuracy

    evaluator = dspy.Evaluate(
        devset=devset, metric=metric, display_progress=display_progress
    )

    try:
        score = evaluator(program)
        logger.info(f"Evaluation score: {score:.3f}")
        return score
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return 0.0


def agreement_rate(program_a, program_b, dataset: List[dspy.Example]) -> float:
    """
    Calculate the agreement rate between two programs on a dataset.

    Args:
        program_a: First program
        program_b: Second program
        dataset: List of examples to evaluate on

    Returns:
        Agreement rate as a float between 0 and 1
    """
    if not dataset:
        logger.warning("Empty dataset provided for agreement calculation")
        return 0.0

    agreements = 0
    total = len(dataset)

    for example in dataset:
        try:
            pred_a = program_a(claim=example.claim)
            pred_b = program_b(claim=example.claim)

            verdict_a = _normalize_verdict(pred_a.verdict)
            verdict_b = _normalize_verdict(pred_b.verdict)

            if verdict_a == verdict_b:
                agreements += 1

        except Exception as e:
            logger.error(f"Error computing agreement for claim '{example.claim}': {e}")
            # Count as disagreement on error
            continue

    rate = agreements / total
    logger.debug(f"Agreement rate: {rate:.3f} ({agreements}/{total})")
    return rate


def detailed_evaluation(program, devset: List[dspy.Example]) -> Dict[str, Any]:
    """
    Perform detailed evaluation with per-example results.

    Args:
        program: The program to evaluate
        devset: List of labeled examples

    Returns:
        Dictionary with detailed results
    """
    results = {
        "total_examples": len(devset),
        "correct": 0,
        "incorrect": 0,
        "errors": 0,
        "accuracy": 0.0,
        "examples": [],
    }

    for i, example in enumerate(devset):
        example_result = {
            "index": i,
            "claim": example.claim,
            "ground_truth": example.verdict if "verdict" in example else None,
            "prediction": None,
            "correct": False,
            "error": None,
        }

        try:
            pred = program(claim=example.claim)
            example_result["prediction"] = pred.verdict

            if "verdict" in example:
                truth_verdict = _normalize_verdict(example.verdict)
                pred_verdict = _normalize_verdict(pred.verdict)

                example_result["correct"] = truth_verdict == pred_verdict

                if example_result["correct"]:
                    results["correct"] += 1
                else:
                    results["incorrect"] += 1

        except Exception as e:
            example_result["error"] = str(e)
            results["errors"] += 1
            logger.error(f"Error evaluating example {i}: {e}")

        results["examples"].append(example_result)

    # Calculate accuracy
    if results["total_examples"] > 0:
        results["accuracy"] = results["correct"] / results["total_examples"]

    logger.info(
        f"Detailed evaluation: {results['correct']}/{results['total_examples']} correct, "
        f"{results['errors']} errors, accuracy={results['accuracy']:.3f}"
    )

    return results


def compare_programs(
    program_a, program_b, devset: List[dspy.Example]
) -> Dict[str, Any]:
    """
    Compare two programs side by side.

    Args:
        program_a: First program
        program_b: Second program
        devset: List of examples to compare on

    Returns:
        Dictionary with comparison results
    """
    results = {
        "total_examples": len(devset),
        "agreement": 0,
        "disagreement": 0,
        "both_correct": 0,
        "both_incorrect": 0,
        "a_correct_b_incorrect": 0,
        "a_incorrect_b_correct": 0,
        "agreement_rate": 0.0,
        "examples": [],
    }

    for i, example in enumerate(devset):
        example_result = {
            "index": i,
            "claim": example.claim,
            "ground_truth": example.verdict if "verdict" in example else None,
            "prediction_a": None,
            "prediction_b": None,
            "agree": False,
            "a_correct": None,
            "b_correct": None,
        }

        try:
            pred_a = program_a(claim=example.claim)
            pred_b = program_b(claim=example.claim)

            example_result["prediction_a"] = pred_a.verdict
            example_result["prediction_b"] = pred_b.verdict

            verdict_a = _normalize_verdict(pred_a.verdict)
            verdict_b = _normalize_verdict(pred_b.verdict)

            # Check agreement
            example_result["agree"] = verdict_a == verdict_b
            if example_result["agree"]:
                results["agreement"] += 1
            else:
                results["disagreement"] += 1

            # Check correctness if ground truth available
            if "verdict" in example:
                truth_verdict = _normalize_verdict(example.verdict)

                example_result["a_correct"] = verdict_a == truth_verdict
                example_result["b_correct"] = verdict_b == truth_verdict

                if example_result["a_correct"] and example_result["b_correct"]:
                    results["both_correct"] += 1
                elif (
                    not example_result["a_correct"] and not example_result["b_correct"]
                ):
                    results["both_incorrect"] += 1
                elif example_result["a_correct"]:
                    results["a_correct_b_incorrect"] += 1
                else:
                    results["a_incorrect_b_correct"] += 1

        except Exception as e:
            logger.error(f"Error comparing programs on example {i}: {e}")

        results["examples"].append(example_result)

    # Calculate agreement rate
    if results["total_examples"] > 0:
        results["agreement_rate"] = results["agreement"] / results["total_examples"]

    logger.info(
        f"Program comparison: {results['agreement']}/{results['total_examples']} agree, "
        f"rate={results['agreement_rate']:.3f}"
    )

    return results
