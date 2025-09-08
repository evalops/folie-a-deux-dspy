"""
Folie à Deux: Iterative LLM Agreement Training

A framework for training two language models to iteratively agree with each other
while maintaining optional anchoring to ground truth.
"""

__version__ = "0.1.0"
__author__ = "EvalOps"
__email__ = "info@evalops.dev"

# Lazy imports to handle missing dependencies gracefully
__all__ = [
    "Verifier",
    "VerifyClaim",
    "create_example",
    "get_dev_labeled",
    "get_train_unlabeled",
    "truth_accuracy",
    "agreement_metric_factory",
    "blended_metric_factory",
    "evaluate",
    "agreement_rate",
    "folie_a_deux",
]


def __getattr__(name):
    """Lazy import mechanism for handling missing dependencies."""
    if name == "Verifier":
        from .verifier import Verifier

        return Verifier
    elif name == "VerifyClaim":
        from .verifier import VerifyClaim

        return VerifyClaim
    elif name == "create_example":
        from .data import create_example

        return create_example
    elif name == "get_dev_labeled":
        from .data import get_dev_labeled

        return get_dev_labeled
    elif name == "get_train_unlabeled":
        from .data import get_train_unlabeled

        return get_train_unlabeled
    elif name == "truth_accuracy":
        from .metrics import truth_accuracy

        return truth_accuracy
    elif name == "agreement_metric_factory":
        from .metrics import agreement_metric_factory

        return agreement_metric_factory
    elif name == "blended_metric_factory":
        from .metrics import blended_metric_factory

        return blended_metric_factory
    elif name == "evaluate":
        from .evaluation import evaluate

        return evaluate
    elif name == "agreement_rate":
        from .evaluation import agreement_rate

        return agreement_rate
    elif name == "folie_a_deux":
        from .experiment import folie_a_deux

        return folie_a_deux
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
