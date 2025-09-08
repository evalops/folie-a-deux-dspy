"""Tests for data module."""

import pytest
from unittest.mock import patch
import dspy
from folie_a_deux.data import (
    create_example,
    get_dev_labeled,
    get_train_unlabeled,
    validate_dataset,
)


class TestCreateExample:
    """Test the create_example function."""

    def test_create_example_with_verdict(self):
        """Test creating example with verdict."""
        example = create_example("Test claim", "yes")

        assert example.claim == "Test claim"
        assert example.verdict == "yes"
        assert "claim" in example.inputs()

    def test_create_example_without_verdict(self):
        """Test creating example without verdict."""
        example = create_example("Test claim")

        assert example.claim == "Test claim"
        assert not hasattr(example, "verdict")
        assert "claim" in example.inputs()

    def test_create_example_none_verdict(self):
        """Test creating example with None verdict."""
        example = create_example("Test claim", None)

        assert example.claim == "Test claim"
        assert not hasattr(example, "verdict")
        assert "claim" in example.inputs()


class TestGetDevLabeled:
    """Test the get_dev_labeled function."""

    def test_get_dev_labeled_size(self):
        """Test dev dataset has correct size."""
        dev_set = get_dev_labeled()
        assert len(dev_set) == 30

    def test_get_dev_labeled_all_have_verdicts(self):
        """Test all dev examples have verdicts."""
        dev_set = get_dev_labeled()

        for example in dev_set:
            assert hasattr(example, "verdict")
            assert example.verdict in ["yes", "no"]
            assert hasattr(example, "claim")
            assert len(example.claim) > 0

    def test_get_dev_labeled_content_samples(self):
        """Test specific content in dev set."""
        dev_set = get_dev_labeled()

        # Check first example
        first = dev_set[0]
        assert "Water boils at 100°C" in first.claim
        assert first.verdict == "yes"

        # Check we have both yes and no examples
        verdicts = [ex.verdict for ex in dev_set]
        assert "yes" in verdicts
        assert "no" in verdicts


class TestGetTrainUnlabeled:
    """Test the get_train_unlabeled function."""

    def test_get_train_unlabeled_default_size(self):
        """Test train dataset has correct default size."""
        train_set = get_train_unlabeled()
        assert len(train_set) == 98  # 7 * 14

    def test_get_train_unlabeled_custom_repetitions(self):
        """Test train dataset with custom repetitions."""
        train_set = get_train_unlabeled(repetitions=3)
        assert len(train_set) == 42  # 3 * 14

    def test_get_train_unlabeled_no_verdicts(self):
        """Test train examples have no verdicts."""
        train_set = get_train_unlabeled()

        for example in train_set:
            assert not hasattr(example, "verdict")
            assert hasattr(example, "claim")
            assert len(example.claim) > 0

    def test_get_train_unlabeled_no_shuffle(self):
        """Test train dataset without shuffling."""
        train_set1 = get_train_unlabeled(shuffle=False)
        train_set2 = get_train_unlabeled(shuffle=False)

        # Should be identical when not shuffled
        claims1 = [ex.claim for ex in train_set1]
        claims2 = [ex.claim for ex in train_set2]
        assert claims1 == claims2

    @patch("folie_a_deux.data.random.shuffle")
    def test_get_train_unlabeled_shuffle_called(self, mock_shuffle):
        """Test that shuffle is called when enabled."""
        get_train_unlabeled(shuffle=True)
        mock_shuffle.assert_called_once()

    def test_get_train_unlabeled_repetitions_work(self):
        """Test that repetitions actually repeat claims."""
        train_set = get_train_unlabeled(repetitions=2, shuffle=False)

        # Should have exactly 2 of each base claim
        claim_counts = {}
        for example in train_set:
            claim = example.claim
            claim_counts[claim] = claim_counts.get(claim, 0) + 1

        # All claims should appear exactly 2 times
        for count in claim_counts.values():
            assert count == 2


class TestValidateDataset:
    """Test the validate_dataset function."""

    def test_validate_empty_dataset(self):
        """Test validation fails on empty dataset."""
        with pytest.raises(ValueError, match="Dataset is empty"):
            validate_dataset([])

    def test_validate_missing_claim_field(self):
        """Test validation fails when claim field missing."""
        invalid_example = dspy.Example(verdict="yes").with_inputs("verdict")

        with pytest.raises(ValueError, match="Example 0 missing 'claim' field"):
            validate_dataset([invalid_example])

    def test_validate_missing_required_verdict(self):
        """Test validation fails when required verdict missing."""
        example = dspy.Example(claim="test").with_inputs("claim")

        with pytest.raises(
            ValueError, match="Example 0 missing required 'verdict' field"
        ):
            validate_dataset([example], require_labels=True)

    def test_validate_invalid_verdict_value(self):
        """Test validation fails on invalid verdict value."""
        example = dspy.Example(claim="test", verdict="maybe").with_inputs("claim")

        with pytest.raises(ValueError, match="Example 0 has invalid verdict: maybe"):
            validate_dataset([example])

    def test_validate_valid_labeled_dataset(self):
        """Test validation passes on valid labeled dataset."""
        examples = [
            dspy.Example(claim="test1", verdict="yes").with_inputs("claim"),
            dspy.Example(claim="test2", verdict="no").with_inputs("claim"),
        ]

        result = validate_dataset(examples, require_labels=True)
        assert result is True

    def test_validate_valid_unlabeled_dataset(self):
        """Test validation passes on valid unlabeled dataset."""
        examples = [
            dspy.Example(claim="test1").with_inputs("claim"),
            dspy.Example(claim="test2").with_inputs("claim"),
        ]

        result = validate_dataset(examples, require_labels=False)
        assert result is True

    def test_validate_mixed_dataset_without_requirement(self):
        """Test validation passes on mixed dataset when labels not required."""
        examples = [
            dspy.Example(claim="test1", verdict="yes").with_inputs("claim"),
            dspy.Example(claim="test2").with_inputs("claim"),  # No verdict
        ]

        result = validate_dataset(examples, require_labels=False)
        assert result is True

    def test_validate_multiple_examples_error_reporting(self):
        """Test validation reports correct example index in errors."""
        examples = [
            dspy.Example(claim="test1", verdict="yes").with_inputs("claim"),
            dspy.Example(claim="test2", verdict="maybe").with_inputs(
                "claim"
            ),  # Invalid
        ]

        with pytest.raises(ValueError, match="Example 1 has invalid verdict: maybe"):
            validate_dataset(examples)


class TestDataIntegration:
    """Integration tests for data functions."""

    def test_dev_and_train_datasets_work_together(self):
        """Test that dev and train datasets can be loaded and validated together."""
        dev_set = get_dev_labeled()
        train_set = get_train_unlabeled()

        # Both should load successfully
        assert len(dev_set) > 0
        assert len(train_set) > 0

        # Both should validate successfully
        validate_dataset(dev_set, require_labels=True)
        validate_dataset(train_set, require_labels=False)

    def test_dataset_consistency(self):
        """Test that datasets are consistent across multiple calls."""
        dev_set1 = get_dev_labeled()
        dev_set2 = get_dev_labeled()

        # Dev set should be identical across calls
        assert len(dev_set1) == len(dev_set2)
        for ex1, ex2 in zip(dev_set1, dev_set2):
            assert ex1.claim == ex2.claim
            assert ex1.verdict == ex2.verdict

    def test_train_set_has_expected_base_claims(self):
        """Test that train set contains expected base claims."""
        train_set = get_train_unlabeled(repetitions=1, shuffle=False)

        # Should have exactly 14 unique claims
        unique_claims = set(ex.claim for ex in train_set)
        assert len(unique_claims) == 14

        # Should contain some expected claims
        claims_text = " ".join(unique_claims)
        assert "Australia" in claims_text
        assert "Shakespeare" in claims_text
        assert "Pacific" in claims_text
