"""Configuration management for Folie à Deux experiments."""

import os
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for Folie à Deux experiments."""

    # Model configuration
    model: str = "ollama_chat/llama3.1:8b"
    api_base: str = "http://localhost:11434"
    api_key: str = ""
    temperature: float = 0.5
    max_tokens: int = 512

    # Experiment parameters
    alpha: float = 0.0  # Truth anchoring weight
    rounds: int = 6  # Number of training rounds
    use_cot: bool = False  # Use Chain of Thought

    # Optimization settings
    auto_mode: str = "light"
    enable_disk_cache: bool = False
    enable_memory_cache: bool = False

    @classmethod
    def from_env(cls) -> "ExperimentConfig":
        """Create configuration from environment variables."""
        return cls(
            model=os.getenv("MODEL", cls.model),
            api_base=os.getenv("API_BASE", cls.api_base),
            api_key=os.getenv("API_KEY", cls.api_key),
            temperature=float(os.getenv("TEMPERATURE", str(cls.temperature))),
            max_tokens=int(os.getenv("MAX_TOKENS", str(cls.max_tokens))),
            alpha=float(os.getenv("ALPHA", str(cls.alpha))),
            rounds=int(os.getenv("ROUNDS", str(cls.rounds))),
            use_cot=os.getenv("USE_COT", "false").lower() == "true",
            auto_mode=os.getenv("AUTO_MODE", cls.auto_mode),
            enable_disk_cache=os.getenv("ENABLE_DISK_CACHE", "false").lower() == "true",
            enable_memory_cache=os.getenv("ENABLE_MEMORY_CACHE", "false").lower()
            == "true",
        )

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"Alpha must be between 0.0 and 1.0, got {self.alpha}")

        if self.rounds < 1:
            raise ValueError(f"Rounds must be >= 1, got {self.rounds}")

        if self.temperature < 0:
            raise ValueError(f"Temperature must be >= 0, got {self.temperature}")

        if self.max_tokens < 1:
            raise ValueError(f"Max tokens must be >= 1, got {self.max_tokens}")

        logger.info(f"Configuration validated: {self}")

    def setup_dspy(self) -> None:
        """Configure DSPy with this configuration."""
        import dspy

        lm = dspy.LM(
            self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        dspy.configure_cache(
            enable_disk_cache=self.enable_disk_cache,
            enable_memory_cache=self.enable_memory_cache,
        )

        dspy.configure(lm=lm)
        logger.info(f"DSPy configured with model: {self.model}")


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,  # Set root level to INFO
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set our application loggers to the requested level
    app_loggers = [
        "folie_a_deux",
        "dspy",
    ]

    for logger_name in app_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))

    # Suppress noisy third-party loggers
    noisy_loggers = [
        "httpcore",
        "httpx",
        "urllib3",
        "requests",
        "LiteLLM Proxy",
        "litellm",
    ]

    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)  # Only show warnings and errors

    # Specifically handle the LiteLLM FastAPI warning
    litellm_logger = logging.getLogger("LiteLLM Proxy")
    litellm_logger.addFilter(
        lambda record: "Missing dependency No module named 'fastapi'"
        not in record.getMessage()
    )
