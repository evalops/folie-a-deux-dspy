"""
Legacy entry point for Folie à Deux experiment.
This file is kept for backward compatibility.
For new usage, prefer: python -m folie_a_deux.main
"""

from folie_a_deux.config import ExperimentConfig, setup_logging
from folie_a_deux.experiment import folie_a_deux


def main():
    """Legacy main function."""
    # Setup logging
    setup_logging()

    # Load configuration from environment
    config = ExperimentConfig.from_env()

    # Run experiment
    result = folie_a_deux(config)

    return result.verifier_a, result.verifier_b, result.rounds


if __name__ == "__main__":
    main()
