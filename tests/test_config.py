"""Tests for configuration module."""

import pytest
import os
from folie_a_deux.config import ExperimentConfig, setup_logging


class TestExperimentConfig:
    """Test the ExperimentConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExperimentConfig()

        assert config.model == "ollama_chat/llama3.1:8b"
        assert config.api_base == "http://localhost:11434"
        assert config.alpha == 0.0
        assert config.rounds == 6
        assert config.use_cot is False
        assert config.temperature == 0.5
        assert config.max_tokens == 512

    def test_from_env_defaults(self):
        """Test configuration from environment with defaults."""
        # Clear relevant env vars
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
            original_values[var] = os.environ.pop(var, None)

        try:
            config = ExperimentConfig.from_env()
            assert config.model == "ollama_chat/llama3.1:8b"
            assert config.alpha == 0.0
            assert config.rounds == 6
        finally:
            # Restore env vars
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value

    def test_from_env_overrides(self):
        """Test configuration from environment with overrides."""
        os.environ["MODEL"] = "test_model"
        os.environ["ALPHA"] = "0.5"
        os.environ["ROUNDS"] = "10"
        os.environ["USE_COT"] = "true"

        try:
            config = ExperimentConfig.from_env()
            assert config.model == "test_model"
            assert config.alpha == 0.5
            assert config.rounds == 10
            assert config.use_cot is True
        finally:
            # Clean up
            for var in ["MODEL", "ALPHA", "ROUNDS", "USE_COT"]:
                os.environ.pop(var, None)

    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        config = ExperimentConfig(alpha=0.5, rounds=5, temperature=0.7, max_tokens=256)
        config.validate()  # Should not raise

    def test_validate_invalid_alpha(self):
        """Test validation with invalid alpha values."""
        with pytest.raises(ValueError, match="Alpha must be between 0.0 and 1.0"):
            config = ExperimentConfig(alpha=-0.1)
            config.validate()

        with pytest.raises(ValueError, match="Alpha must be between 0.0 and 1.0"):
            config = ExperimentConfig(alpha=1.5)
            config.validate()

    def test_validate_invalid_rounds(self):
        """Test validation with invalid rounds."""
        with pytest.raises(ValueError, match="Rounds must be >= 1"):
            config = ExperimentConfig(rounds=0)
            config.validate()

    def test_validate_invalid_temperature(self):
        """Test validation with invalid temperature."""
        with pytest.raises(ValueError, match="Temperature must be >= 0"):
            config = ExperimentConfig(temperature=-0.1)
            config.validate()

    def test_validate_invalid_max_tokens(self):
        """Test validation with invalid max_tokens."""
        with pytest.raises(ValueError, match="Max tokens must be >= 1"):
            config = ExperimentConfig(max_tokens=0)
            config.validate()


def test_setup_logging():
    """Test logging setup function."""
    # Test default level
    setup_logging()

    # Test custom level
    setup_logging("DEBUG")
    setup_logging("WARNING")
