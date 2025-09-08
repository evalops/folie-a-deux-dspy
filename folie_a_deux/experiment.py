"""Main experiment implementation for Folie à Deux."""

from typing import List, Dict, Tuple, Any
import logging
from dspy.teleprompt import MIPROv2

from .config import ExperimentConfig
from .verifier import Verifier
from .data import get_dev_labeled, get_train_unlabeled, validate_dataset
from .metrics import truth_accuracy, agreement_metric_factory, blended_metric_factory
from .evaluation import evaluate, agreement_rate

logger = logging.getLogger(__name__)


class ExperimentResults:
    """Container for experiment results."""

    def __init__(self):
        self.rounds: List[Dict[str, Any]] = []
        self.verifier_a = None
        self.verifier_b = None

    def add_round(
        self,
        round_num: int,
        acc_a: float,
        acc_b: float,
        agree_dev: float,
        agree_train: float,
    ) -> None:
        """Add results for a training round."""
        self.rounds.append(
            {
                "round": round_num,
                "accuracy_a": acc_a,
                "accuracy_b": acc_b,
                "agreement_dev": agree_dev,
                "agreement_train": agree_train,
            }
        )

    def get_final_accuracies(self) -> Tuple[float, float]:
        """Get final accuracies for both verifiers."""
        if not self.rounds:
            return 0.0, 0.0
        last_round = self.rounds[-1]
        return last_round["accuracy_a"], last_round["accuracy_b"]

    def get_final_agreement(self) -> float:
        """Get final agreement rate on dev set."""
        if not self.rounds:
            return 0.0
        return self.rounds[-1]["agreement_dev"]

    def summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        if not self.rounds:
            return {"total_rounds": 0}

        acc_a, acc_b = self.get_final_accuracies()
        final_agreement = self.get_final_agreement()

        return {
            "total_rounds": len(self.rounds),
            "final_accuracy_a": acc_a,
            "final_accuracy_b": acc_b,
            "final_agreement": final_agreement,
            "max_accuracy_a": max(r["accuracy_a"] for r in self.rounds),
            "max_accuracy_b": max(r["accuracy_b"] for r in self.rounds),
            "max_agreement": max(r["agreement_dev"] for r in self.rounds),
        }


def folie_a_deux(
    config: ExperimentConfig = None,
    rounds: int = None,
    use_cot: bool = None,
    alpha_anchor: float = None,
) -> ExperimentResults:
    """
    Run the Folie à Deux experiment.

    Args:
        config: Experiment configuration (if None, loads from environment)
        rounds: Override number of rounds
        use_cot: Override Chain of Thought setting
        alpha_anchor: Override truth anchoring weight

    Returns:
        ExperimentResults object containing all results
    """
    # Setup configuration
    if config is None:
        config = ExperimentConfig.from_env()

    # Apply overrides
    if rounds is not None:
        config.rounds = rounds
    if use_cot is not None:
        config.use_cot = use_cot
    if alpha_anchor is not None:
        config.alpha = alpha_anchor

    config.validate()
    config.setup_dspy()

    logger.info(f"Starting Folie à Deux experiment with {config.rounds} rounds")
    logger.info(f"Alpha (truth anchoring): {config.alpha}")
    logger.info(f"Chain of Thought: {config.use_cot}")

    # Load and validate datasets
    dev_labeled = get_dev_labeled()
    train_unlabeled = get_train_unlabeled()

    validate_dataset(dev_labeled, require_labels=True)
    validate_dataset(train_unlabeled, require_labels=False)

    logger.info(
        f"Loaded {len(dev_labeled)} labeled examples, {len(train_unlabeled)} unlabeled"
    )

    # Initialize verifiers
    verifier_a = Verifier(use_cot=config.use_cot)
    verifier_b = Verifier(use_cot=config.use_cot)

    results = ExperimentResults()

    try:
        # Baseline: truth-optimized verifier A
        logger.info("Training baseline verifier A on ground truth...")
        optimizer_a = MIPROv2(metric=truth_accuracy, auto=config.auto_mode)
        verifier_a = optimizer_a.compile(verifier_a, trainset=dev_labeled)

        # Initialize verifier B with same baseline training
        logger.info("Training baseline verifier B on ground truth...")
        optimizer_b = MIPROv2(metric=truth_accuracy, auto=config.auto_mode)
        verifier_b = optimizer_b.compile(verifier_b, trainset=dev_labeled)

        # Iterative co-training
        for round_num in range(1, config.rounds + 1):
            logger.info(f"Starting round {round_num}/{config.rounds}")

            # Choose metric based on alpha value
            if config.alpha > 0:
                metric_a = blended_metric_factory(verifier_b, config.alpha)
                metric_b = blended_metric_factory(verifier_a, config.alpha)
                logger.debug(f"Using blended metric with alpha={config.alpha}")
            else:
                metric_a = agreement_metric_factory(verifier_b)
                metric_b = agreement_metric_factory(verifier_a)
                logger.debug("Using pure agreement metric")

            # Train A to agree with B
            logger.debug("Training verifier A...")
            optimizer_a = MIPROv2(metric=metric_a, auto=config.auto_mode)
            verifier_a = optimizer_a.compile(verifier_a, trainset=train_unlabeled)

            # Train B to agree with A
            logger.debug("Training verifier B...")
            optimizer_b = MIPROv2(metric=metric_b, auto=config.auto_mode)
            verifier_b = optimizer_b.compile(verifier_b, trainset=train_unlabeled)

            # Evaluate both verifiers
            acc_a = evaluate(verifier_a, dev_labeled, display_progress=False)
            acc_b = evaluate(verifier_b, dev_labeled, display_progress=False)

            # Calculate agreement rates
            agree_dev = agreement_rate(verifier_a, verifier_b, dev_labeled)
            # Use min to avoid hardcoded slice that could fail with small datasets
            train_sample_size = min(60, len(train_unlabeled))
            agree_train = agreement_rate(
                verifier_a, verifier_b, train_unlabeled[:train_sample_size]
            )

            # Store results
            results.add_round(round_num, acc_a, acc_b, agree_dev, agree_train)

            # Log progress
            logger.info(
                f"[round {round_num}] accA={acc_a:.3f} accB={acc_b:.3f} "
                f"agree_dev={agree_dev:.3f} agree_train={agree_train:.3f}"
            )

            print(
                f"[round {round_num}] accA={acc_a:.3f} accB={acc_b:.3f} "
                f"agree_dev={agree_dev:.3f} agree_train={agree_train:.3f}"
            )

        # Store final verifiers
        results.verifier_a = verifier_a
        results.verifier_b = verifier_b

        logger.info("Experiment completed successfully")
        logger.info(f"Final summary: {results.summary()}")

        return results

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


def run_ablation_study(
    base_config: ExperimentConfig = None,
) -> Dict[str, ExperimentResults]:
    """
    Run an ablation study with different alpha values.

    Args:
        base_config: Base configuration to modify

    Returns:
        Dictionary mapping alpha values to results
    """
    if base_config is None:
        base_config = ExperimentConfig.from_env()

    alpha_values = [0.0, 0.05, 0.1, 0.2, 0.5]
    results = {}

    logger.info(f"Running ablation study with alpha values: {alpha_values}")

    for alpha in alpha_values:
        logger.info(f"Running experiment with alpha={alpha}")

        # Create config copy with modified alpha
        config = ExperimentConfig(
            model=base_config.model,
            api_base=base_config.api_base,
            api_key=base_config.api_key,
            temperature=base_config.temperature,
            max_tokens=base_config.max_tokens,
            alpha=alpha,
            rounds=base_config.rounds,
            use_cot=base_config.use_cot,
            auto_mode=base_config.auto_mode,
            enable_disk_cache=base_config.enable_disk_cache,
            enable_memory_cache=base_config.enable_memory_cache,
        )

        try:
            results[f"alpha_{alpha}"] = folie_a_deux(config)
        except Exception as e:
            logger.error(f"Failed experiment with alpha={alpha}: {e}")
            results[f"alpha_{alpha}"] = None

    logger.info("Ablation study completed")
    return results
