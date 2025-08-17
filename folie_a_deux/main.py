"""Main entry point for the Folie à Deux experiment."""

import argparse
import sys
import json
from typing import Optional

from .config import ExperimentConfig, setup_logging
from .experiment import folie_a_deux, run_ablation_study


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: Optional command line arguments (for testing)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Folie à Deux: Iterative LLM Agreement Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment parameters
    parser.add_argument(
        "--rounds", type=int, default=None, help="Number of training rounds"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Truth anchoring weight (0=pure agreement, 1=pure truth)",
    )
    parser.add_argument(
        "--use-cot", action="store_true", help="Use Chain of Thought reasoning"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (e.g., ollama_chat/llama3.1:8b)",
    )
    parser.add_argument("--api-base", type=str, default=None, help="API base URL")
    parser.add_argument(
        "--temperature", type=float, default=None, help="Model temperature"
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens")

    # Experiment modes
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study with different alpha values",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for results (JSON format)"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    # Parse arguments
    parsed_args = parser.parse_args(args)

    # Setup logging
    setup_logging(parsed_args.log_level)

    try:
        # Load configuration from environment
        config = ExperimentConfig.from_env()

        # Apply command line overrides
        if parsed_args.model:
            config.model = parsed_args.model
        if parsed_args.api_base:
            config.api_base = parsed_args.api_base
        if parsed_args.temperature is not None:
            config.temperature = parsed_args.temperature
        if parsed_args.max_tokens is not None:
            config.max_tokens = parsed_args.max_tokens
        if parsed_args.rounds is not None:
            config.rounds = parsed_args.rounds
        if parsed_args.alpha is not None:
            config.alpha = parsed_args.alpha

        # Run experiment
        if parsed_args.ablation:
            print("Running ablation study...")
            results = run_ablation_study(config)

            # Convert results to serializable format
            output_data = {}
            for key, result in results.items():
                if result is not None:
                    output_data[key] = {
                        "summary": result.summary(),
                        "rounds": result.rounds,
                    }
                else:
                    output_data[key] = None

        else:
            print("Running single experiment...")
            result = folie_a_deux(config=config, use_cot=parsed_args.use_cot)

            output_data = {"summary": result.summary(), "rounds": result.rounds}

        # Save results if output file specified
        if parsed_args.output:
            with open(parsed_args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Results saved to {parsed_args.output}")

        print("\nExperiment completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
