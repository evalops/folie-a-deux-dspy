"""Verifier module containing the core DSPy components."""

import random
import logging
import dspy

logger = logging.getLogger(__name__)


class VerifyClaim(dspy.Signature):
    """Decide if a claim is factually correct. Output strictly 'yes' or 'no'."""

    claim: str = dspy.InputField()
    verdict: str = dspy.OutputField(desc="either 'yes' or 'no'")


class Verifier(dspy.Module):
    """A factual claim verifier using DSPy."""

    def __init__(self, use_cot: bool = False):
        """
        Initialize the verifier.

        Args:
            use_cot: Whether to use Chain of Thought reasoning
        """
        super().__init__()
        self.use_cot = use_cot
        self.step = (dspy.ChainOfThought if use_cot else dspy.Predict)(VerifyClaim)
        logger.debug(f"Initialized Verifier with use_cot={use_cot}")

    def forward(self, claim: str) -> dspy.Prediction:
        """
        Verify a factual claim.

        Args:
            claim: The claim to verify

        Returns:
            Prediction with normalized verdict ('yes' or 'no')
        """
        logger.debug(f"Verifying claim: {claim}")

        try:
            out = self.step(claim=claim)

            # Normalize verdict to ensure it's exactly 'yes' or 'no'
            raw_verdict = (out.verdict or "").strip().lower()

            if "yes" in raw_verdict and "no" not in raw_verdict:
                normalized_verdict = "yes"
            elif "no" in raw_verdict and "yes" not in raw_verdict:
                normalized_verdict = "no"
            else:
                # Ambiguous response - log and choose randomly
                logger.warning(f"Ambiguous verdict '{out.verdict}' for claim: {claim}")
                normalized_verdict = random.choice(["yes", "no"])

            out.verdict = normalized_verdict
            logger.debug(f"Normalized verdict: {normalized_verdict}")

            return out

        except Exception as e:
            logger.error(f"Error verifying claim '{claim}': {e}")
            # Return a random verdict on error
            out = dspy.Prediction(verdict=random.choice(["yes", "no"]))
            return out
