"""Tests for verifier module."""

import pytest
from unittest.mock import Mock, patch
from folie_a_deux.verifier import VerifyClaim, Verifier


class TestVerifyClaim:
    """Test the VerifyClaim DSPy signature."""

    def test_verify_claim_signature(self):
        """Test VerifyClaim signature has correct fields."""
        # Test that the signature has the expected fields
        assert hasattr(VerifyClaim, "__annotations__")
        annotations = VerifyClaim.__annotations__
        assert "claim" in annotations
        assert "verdict" in annotations

        # Test the signature can be instantiated with required fields
        signature = VerifyClaim(claim="test", verdict="yes")
        assert signature is not None
        assert signature.claim == "test"
        assert signature.verdict == "yes"

    def test_verify_claim_docstring(self):
        """Test VerifyClaim has proper docstring."""
        assert "factually correct" in VerifyClaim.__doc__
        assert "yes" in VerifyClaim.__doc__
        assert "no" in VerifyClaim.__doc__


class TestVerifier:
    """Test the Verifier DSPy module."""

    def test_verifier_init_default(self):
        """Test Verifier initialization with default parameters."""
        with patch("dspy.Predict") as mock_predict:
            verifier = Verifier()

            assert verifier.use_cot is False
            mock_predict.assert_called_once_with(VerifyClaim)

    def test_verifier_init_with_cot(self):
        """Test Verifier initialization with Chain of Thought."""
        with patch("dspy.ChainOfThought") as mock_cot:
            verifier = Verifier(use_cot=True)

            assert verifier.use_cot is True
            mock_cot.assert_called_once_with(VerifyClaim)

    @patch("dspy.Predict")
    def test_verifier_forward_success(self, mock_predict):
        """Test successful verdict prediction."""
        # Mock the prediction step
        mock_step = Mock()
        mock_prediction = Mock()
        mock_prediction.verdict = "yes"
        mock_step.return_value = mock_prediction
        mock_predict.return_value = mock_step

        verifier = Verifier()
        verifier.step = mock_step

        result = verifier.forward("Test claim")

        assert result.verdict == "yes"
        mock_step.assert_called_once_with(claim="Test claim")

    @patch("dspy.Predict")
    def test_verifier_forward_normalize_yes(self, mock_predict):
        """Test verdict normalization for yes responses."""
        mock_step = Mock()
        mock_predict.return_value = mock_step

        verifier = Verifier()
        verifier.step = mock_step

        # Test various yes formats
        test_cases = ["YES", "Yes", "yes definitely", "I think yes", "  yes  "]

        for test_verdict in test_cases:
            mock_prediction = Mock()
            mock_prediction.verdict = test_verdict
            mock_step.return_value = mock_prediction

            result = verifier.forward("Test claim")
            assert result.verdict == "yes", f"Failed for input: {test_verdict}"

    @patch("dspy.Predict")
    def test_verifier_forward_normalize_no(self, mock_predict):
        """Test verdict normalization for no responses."""
        mock_step = Mock()
        mock_predict.return_value = mock_step

        verifier = Verifier()
        verifier.step = mock_step

        # Test various no formats
        test_cases = ["NO", "No", "no way", "definitely no", "  no  "]

        for test_verdict in test_cases:
            mock_prediction = Mock()
            mock_prediction.verdict = test_verdict
            mock_step.return_value = mock_prediction

            result = verifier.forward("Test claim")
            assert result.verdict == "no", f"Failed for input: {test_verdict}"

    @patch("dspy.Predict")
    @patch("folie_a_deux.verifier.random.choice")
    def test_verifier_forward_ambiguous_response(self, mock_choice, mock_predict):
        """Test handling of ambiguous responses."""
        mock_step = Mock()
        mock_predict.return_value = mock_step
        mock_choice.return_value = "yes"

        verifier = Verifier()
        verifier.step = mock_step

        # Test ambiguous responses
        test_cases = ["maybe", "yes and no", "both yes and no", "unclear", ""]

        for test_verdict in test_cases:
            mock_prediction = Mock()
            mock_prediction.verdict = test_verdict
            mock_step.return_value = mock_prediction

            result = verifier.forward("Test claim")
            assert result.verdict in ["yes", "no"], f"Failed for input: {test_verdict}"

        # Should have called random.choice for each ambiguous case
        assert mock_choice.call_count == len(test_cases)

    @patch("dspy.Predict")
    @patch("folie_a_deux.verifier.random.choice")
    def test_verifier_forward_none_verdict(self, mock_choice, mock_predict):
        """Test handling of None verdict."""
        mock_step = Mock()
        mock_predict.return_value = mock_step
        mock_choice.return_value = "no"

        verifier = Verifier()
        verifier.step = mock_step

        mock_prediction = Mock()
        mock_prediction.verdict = None
        mock_step.return_value = mock_prediction

        result = verifier.forward("Test claim")
        assert result.verdict in ["yes", "no"]
        mock_choice.assert_called_once_with(["yes", "no"])

    @patch("dspy.Predict")
    @patch("folie_a_deux.verifier.random.choice")
    def test_verifier_forward_exception_handling(self, mock_choice, mock_predict):
        """Test exception handling in forward method."""
        mock_step = Mock()
        mock_predict.return_value = mock_step
        mock_choice.return_value = "yes"

        # Make the step raise an exception
        mock_step.side_effect = Exception("Test error")

        verifier = Verifier()
        verifier.step = mock_step

        result = verifier.forward("Test claim")

        # Should return a prediction with random verdict
        assert hasattr(result, "verdict")
        assert result.verdict in ["yes", "no"]
        mock_choice.assert_called_once_with(["yes", "no"])

    @patch("dspy.Predict")
    def test_verifier_forward_both_yes_and_no(self, mock_predict):
        """Test handling when response contains both yes and no."""
        mock_step = Mock()
        mock_predict.return_value = mock_step

        verifier = Verifier()
        verifier.step = mock_step

        # Test responses with both yes and no
        test_cases = ["yes but also no", "no, wait yes", "both yes and no"]

        with patch("folie_a_deux.verifier.random.choice", return_value="yes"):
            for test_verdict in test_cases:
                mock_prediction = Mock()
                mock_prediction.verdict = test_verdict
                mock_step.return_value = mock_prediction

                result = verifier.forward("Test claim")
                # Should use random choice when both are present
                assert result.verdict in ["yes", "no"]

    @patch("dspy.Predict")
    def test_verifier_forward_only_yes_with_no_mentioned(self, mock_predict):
        """Test responses with yes primary but no mentioned."""
        mock_step = Mock()
        mock_predict.return_value = mock_step

        verifier = Verifier()
        verifier.step = mock_step

        # Cases where yes is primary but no is mentioned
        mock_prediction = Mock()
        mock_prediction.verdict = "not no, but yes"  # yes primary
        mock_step.return_value = mock_prediction

        with patch("folie_a_deux.verifier.random.choice", return_value="yes"):
            result = verifier.forward("Test claim")
            # Should be ambiguous since both are present
            assert result.verdict in ["yes", "no"]

    @patch("dspy.Predict")
    def test_verifier_forward_only_no_with_yes_mentioned(self, mock_predict):
        """Test responses with no primary but yes mentioned."""
        mock_step = Mock()
        mock_predict.return_value = mock_step

        verifier = Verifier()
        verifier.step = mock_step

        # Cases where no is primary but yes is mentioned
        mock_prediction = Mock()
        mock_prediction.verdict = "not yes, definitely no"  # no primary
        mock_step.return_value = mock_prediction

        with patch("folie_a_deux.verifier.random.choice", return_value="yes"):
            result = verifier.forward("Test claim")
            # Should be ambiguous since both are present
            assert result.verdict in ["yes", "no"]


class TestVerifierIntegration:
    """Integration tests for verifier components."""

    @patch("dspy.Predict")
    def test_verifier_can_be_created_and_called(self, mock_predict):
        """Test that verifier can be created and called without errors."""
        mock_step = Mock()
        mock_prediction = Mock()
        mock_prediction.verdict = "yes"
        mock_step.return_value = mock_prediction
        mock_predict.return_value = mock_step

        # Test both CoT and regular verifiers
        for use_cot in [True, False]:
            verifier = Verifier(use_cot=use_cot)
            verifier.step = mock_step

            result = verifier.forward("Test claim")
            assert hasattr(result, "verdict")
            assert result.verdict in ["yes", "no"]

    def test_verifier_signature_compatibility(self):
        """Test that VerifyClaim signature is compatible with DSPy expectations."""
        # This tests the signature can be instantiated properly
        try:
            # This should not raise an error
            signature = VerifyClaim
            assert signature is not None

            # Test that it has the expected structure
            assert hasattr(signature, "__doc__")
            assert signature.__doc__ is not None
        except Exception as e:
            pytest.fail(f"VerifyClaim signature incompatible with DSPy: {e}")

    @patch("dspy.ChainOfThought")
    @patch("dspy.Predict")
    def test_verifier_uses_correct_dspy_components(self, mock_predict, mock_cot):
        """Test that verifier uses correct DSPy components based on use_cot."""
        # Test without CoT
        Verifier(use_cot=False)
        mock_predict.assert_called_with(VerifyClaim)

        # Test with CoT
        Verifier(use_cot=True)
        mock_cot.assert_called_with(VerifyClaim)
