"""Data management for factual claims."""

import random
from typing import List, Optional
import dspy
import logging

logger = logging.getLogger(__name__)


def create_example(claim: str, verdict: Optional[str] = None) -> dspy.Example:
    """
    Create a DSPy example for a claim.

    Args:
        claim: The factual claim
        verdict: Optional ground truth verdict ('yes' or 'no')

    Returns:
        DSPy Example with inputs configured
    """
    ex = dspy.Example(claim=claim)
    if verdict is not None:
        ex["verdict"] = verdict
    return ex.with_inputs("claim")


def get_dev_labeled() -> List[dspy.Example]:
    """Get the labeled development dataset for evaluation."""
    return [
        create_example("Water boils at 100°C at sea level.", "yes"),
        create_example("The capital of Australia is Sydney.", "no"),
        create_example("Electrons are larger than atoms.", "no"),
        create_example(
            "The Great Wall is visible from space with the naked eye.", "no"
        ),
        create_example("Shakespeare wrote 'Hamlet'.", "yes"),
        create_example("Bananas grow on trees.", "no"),
        create_example("The sun is a star.", "yes"),
        create_example("There are 7 continents on Earth.", "yes"),
        create_example("Lightning never strikes the same place twice.", "no"),
        create_example("Coffee is made from beans that are actually seeds.", "yes"),
        create_example("Humans have more than 5 senses.", "yes"),
        create_example("The speed of light in vacuum is ~3e8 m/s.", "yes"),
        create_example("The Nile is the longest river in the world.", "no"),
        create_example("Gold has chemical symbol Au.", "yes"),
        create_example("Bulls are enraged by the color red.", "no"),
        create_example("The capital of Canada is Toronto.", "no"),
        create_example("The Pacific is the largest ocean.", "yes"),
        create_example("Tomatoes are fruits (botanically).", "yes"),
        create_example("A koala is a bear.", "no"),
        create_example("Mount Everest is over 8000 meters.", "yes"),
        create_example("The human body has 206 bones in adulthood.", "yes"),
        create_example("Penguins live in the Arctic wild.", "no"),
        create_example("Sound travels faster than light.", "no"),
        create_example("Helium is lighter than air.", "yes"),
        create_example("The capital of Brazil is Rio de Janeiro.", "no"),
        create_example("Pluto is classified as a dwarf planet.", "yes"),
        create_example("Bamboo is a type of grass.", "yes"),
        create_example("An octagon has 9 sides.", "no"),
        create_example("Canberra is Australia's capital.", "yes"),
        create_example("The Mona Lisa was painted by da Vinci.", "yes"),
    ]


def get_train_unlabeled(
    repetitions: int = 7, shuffle: bool = True
) -> List[dspy.Example]:
    """
    Get the unlabeled training dataset for agreement optimization.

    Args:
        repetitions: Number of times to repeat the base claims
        shuffle: Whether to shuffle the final dataset

    Returns:
        List of unlabeled examples
    """
    base_claims = [
        "The capital of Australia is Sydney.",
        "Bananas grow on trees.",
        "The moon has a permanent dark side.",
        "Gold's symbol is Au.",
        "There are 5 continents.",
        "Tomatoes are vegetables (botanically).",
        "Coffee beans are seeds.",
        "Electrons are bigger than atoms.",
        "The Great Wall can be seen from space unaided.",
        "Helium is heavier than air.",
        "Lightning avoids tall buildings.",
        "Shakespeare wrote Hamlet.",
        "The Pacific is the biggest ocean.",
        "Bulls hate red.",
    ]

    # Create unlabeled examples with repetitions
    train_unlabeled = [create_example(claim) for claim in base_claims] * repetitions

    if shuffle:
        random.shuffle(train_unlabeled)

    logger.info(f"Created {len(train_unlabeled)} unlabeled training examples")
    return train_unlabeled


def validate_dataset(dataset: List[dspy.Example], require_labels: bool = False) -> bool:
    """
    Validate a dataset for consistency.

    Args:
        dataset: List of examples to validate
        require_labels: Whether to require verdict labels

    Returns:
        True if dataset is valid

    Raises:
        ValueError: If dataset is invalid
    """
    if not dataset:
        raise ValueError("Dataset is empty")

    for i, example in enumerate(dataset):
        if "claim" not in example:
            raise ValueError(f"Example {i} missing 'claim' field")

        if require_labels and "verdict" not in example:
            raise ValueError(f"Example {i} missing required 'verdict' field")

        if "verdict" in example and example.verdict not in ["yes", "no"]:
            raise ValueError(f"Example {i} has invalid verdict: {example.verdict}")

    logger.info(f"Validated dataset with {len(dataset)} examples")
    return True
