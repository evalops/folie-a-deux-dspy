"""Pytest configuration and fixtures."""

import pytest
import os
from unittest.mock import Mock
import dspy
from folie_a_deux.config import ExperimentConfig


@pytest.fixture
def sample_config():
    """Provide a sample experiment configuration."""
    return ExperimentConfig(
        model="test_model",
        alpha=0.1,
        rounds=3,
        use_cot=False,
        temperature=0.5,
        max_tokens=256,
    )


@pytest.fixture
def sample_examples():
    """Provide sample DSPy examples for testing."""
    return [
        dspy.Example(claim="The Earth is round", verdict="true"),
        dspy.Example(claim="The sky is green", verdict="false"),
        dspy.Example(claim="Water boils at 100°C", verdict="true"),
    ]


@pytest.fixture
def sample_unlabeled_examples():
    """Provide sample unlabeled DSPy examples for testing."""
    return [
        dspy.Example(claim="The moon orbits Earth"),
        dspy.Example(claim="Fish can fly"),
        dspy.Example(claim="Ice is frozen water"),
    ]


@pytest.fixture
def mock_verifier():
    """Provide a mock verifier."""
    verifier = Mock()

    # Mock prediction
    mock_pred = Mock()
    mock_pred.verdict = "true"
    verifier.return_value = mock_pred

    return verifier


@pytest.fixture
def mock_optimizer():
    """Provide a mock MIPROv2 optimizer."""
    optimizer = Mock()

    # Mock the compile method to return the input program
    def compile_side_effect(program, trainset=None):
        return program

    optimizer.compile.side_effect = compile_side_effect
    return optimizer


@pytest.fixture
def clean_env():
    """Clean environment variables before and after test."""
    # Store original values
    env_vars = [
        "MODEL",
        "API_BASE",
        "ALPHA",
        "ROUNDS",
        "USE_COT",
        "TEMPERATURE",
        "MAX_TOKENS",
    ]
    original_values = {}

    for var in env_vars:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original values
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


@pytest.fixture(autouse=True)
def mock_dspy_configure():
    """Mock DSPy configuration to avoid actual LLM calls."""
    with pytest.MonkeyPatch().context() as m:
        # Mock DSPy LM and configure functions
        m.setattr("dspy.LM", Mock)
        m.setattr("dspy.configure", Mock)
        m.setattr("dspy.configure_cache", Mock)
        yield


@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing."""
    import logging

    logger = Mock(spec=logging.Logger)
    return logger
