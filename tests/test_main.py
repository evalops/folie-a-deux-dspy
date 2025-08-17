"""Tests for main CLI module."""

import pytest
import json
from unittest.mock import patch, Mock, mock_open
from folie_a_deux.main import main
from folie_a_deux.config import ExperimentConfig


class TestMainCLI:
    """Test the main CLI interface."""

    @patch("folie_a_deux.main.folie_a_deux")
    @patch("folie_a_deux.main.setup_logging")
    def test_main_default_run(self, mock_setup_logging, mock_folie_a_deux):
        """Test main with default arguments."""
        mock_result = Mock()
        mock_result.summary.return_value = {"test": "result"}
        mock_result.rounds = [{"round": 1}]
        mock_folie_a_deux.return_value = mock_result

        exit_code = main([])

        assert exit_code == 0
        mock_setup_logging.assert_called_once_with("INFO")
        mock_folie_a_deux.assert_called_once()

    @patch("folie_a_deux.main.folie_a_deux")
    @patch("folie_a_deux.main.setup_logging")
    def test_main_with_custom_arguments(self, mock_setup_logging, mock_folie_a_deux):
        """Test main with custom arguments."""
        mock_result = Mock()
        mock_result.summary.return_value = {"test": "result"}
        mock_result.rounds = [{"round": 1}]
        mock_folie_a_deux.return_value = mock_result

        args = [
            "--rounds",
            "5",
            "--alpha",
            "0.3",
            "--model",
            "test_model",
            "--temperature",
            "0.8",
            "--max-tokens",
            "256",
            "--use-cot",
            "--log-level",
            "DEBUG",
        ]

        exit_code = main(args)

        assert exit_code == 0
        mock_setup_logging.assert_called_once_with("DEBUG")

        # Check that folie_a_deux was called with correct config
        call_args = mock_folie_a_deux.call_args
        config = call_args[1]["config"]
        assert config.rounds == 5
        assert config.alpha == 0.3
        assert config.model == "test_model"
        assert config.temperature == 0.8
        assert config.max_tokens == 256
        assert call_args[1]["use_cot"] is True

    @patch("folie_a_deux.main.run_ablation_study")
    @patch("folie_a_deux.main.setup_logging")
    def test_main_ablation_study(self, mock_setup_logging, mock_ablation):
        """Test main with ablation study."""
        mock_results = {
            "alpha_0.0": Mock(),
            "alpha_0.1": Mock(),
        }
        for key, mock_result in mock_results.items():
            mock_result.summary.return_value = {"alpha": key}
            mock_result.rounds = [{"round": 1}]
        mock_ablation.return_value = mock_results

        args = ["--ablation", "--rounds", "3"]

        exit_code = main(args)

        assert exit_code == 0
        mock_ablation.assert_called_once()

        # Check that ablation was called with correct config
        call_args = mock_ablation.call_args[0]
        config = call_args[0]
        assert config.rounds == 3

    @patch("folie_a_deux.main.folie_a_deux")
    @patch("folie_a_deux.main.setup_logging")
    @patch("builtins.open", new_callable=mock_open)
    def test_main_with_output_file(
        self, mock_file, mock_setup_logging, mock_folie_a_deux
    ):
        """Test main with output file."""
        mock_result = Mock()
        mock_result.summary.return_value = {"accuracy": 0.85}
        mock_result.rounds = [{"round": 1, "acc": 0.8}]
        mock_folie_a_deux.return_value = mock_result

        args = ["--output", "results.json", "--rounds", "2"]

        exit_code = main(args)

        assert exit_code == 0

        # Check that file was opened for writing
        mock_file.assert_called_once_with("results.json", "w")

        # Check that JSON was written
        handle = mock_file()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)

        # Parse the written JSON to verify it's valid
        try:
            data = json.loads(written_content)
            assert "summary" in data
            assert "rounds" in data
        except json.JSONDecodeError:
            pytest.fail("Invalid JSON written to output file")

    @patch("folie_a_deux.main.run_ablation_study")
    @patch("folie_a_deux.main.setup_logging")
    @patch("builtins.open", new_callable=mock_open)
    def test_main_ablation_with_output(
        self, mock_file, mock_setup_logging, mock_ablation
    ):
        """Test ablation study with output file."""
        mock_results = {
            "alpha_0.0": Mock(),
            "alpha_0.1": None,  # Test with some failed experiments
        }
        mock_results["alpha_0.0"].summary.return_value = {"alpha": "0.0"}
        mock_results["alpha_0.0"].rounds = [{"round": 1}]
        mock_ablation.return_value = mock_results

        args = ["--ablation", "--output", "ablation_results.json"]

        exit_code = main(args)

        assert exit_code == 0
        mock_file.assert_called_once_with("ablation_results.json", "w")

        # Check that JSON was written
        handle = mock_file()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)

        # Parse and verify the ablation results
        try:
            data = json.loads(written_content)
            assert "alpha_0.0" in data
            assert "alpha_0.1" in data
            assert data["alpha_0.0"] is not None
            assert data["alpha_0.1"] is None
        except json.JSONDecodeError:
            pytest.fail("Invalid JSON written to ablation output file")

    @patch("folie_a_deux.main.setup_logging")
    def test_main_keyboard_interrupt(self, mock_setup_logging):
        """Test main handles KeyboardInterrupt gracefully."""
        with patch(
            "folie_a_deux.main.ExperimentConfig.from_env", side_effect=KeyboardInterrupt
        ):
            exit_code = main([])
            assert exit_code == 1

    @patch("folie_a_deux.main.setup_logging")
    def test_main_general_exception(self, mock_setup_logging):
        """Test main handles general exceptions gracefully."""
        with patch(
            "folie_a_deux.main.ExperimentConfig.from_env",
            side_effect=ValueError("Test error"),
        ):
            exit_code = main([])
            assert exit_code == 1

    def test_main_argument_parsing(self):
        """Test argument parsing with various combinations."""
        # Test with quiet flag
        with (
            patch("folie_a_deux.main.folie_a_deux") as mock_exp,
            patch("folie_a_deux.main.setup_logging") as mock_log,
        ):
            mock_result = Mock()
            mock_result.summary.return_value = {}
            mock_result.rounds = []
            mock_exp.return_value = mock_result

            exit_code = main(["--quiet", "--alpha", "0.5"])
            assert exit_code == 0

    @patch("folie_a_deux.main.folie_a_deux")
    @patch("folie_a_deux.main.setup_logging")
    def test_main_environment_overrides(self, mock_setup_logging, mock_folie_a_deux):
        """Test that command line args override environment config."""
        mock_result = Mock()
        mock_result.summary.return_value = {"test": "result"}
        mock_result.rounds = [{"round": 1}]
        mock_folie_a_deux.return_value = mock_result

        # Mock environment config
        with patch("folie_a_deux.main.ExperimentConfig.from_env") as mock_from_env:
            env_config = ExperimentConfig(alpha=0.1, rounds=3, model="env_model")
            mock_from_env.return_value = env_config

            # Override with command line args
            args = ["--alpha", "0.7", "--rounds", "8"]
            exit_code = main(args)

            assert exit_code == 0

            # Check that the config was modified by CLI args
            call_args = mock_folie_a_deux.call_args
            config = call_args[1]["config"]
            assert config.alpha == 0.7  # Overridden
            assert config.rounds == 8  # Overridden
            assert config.model == "env_model"  # From environment

    @patch("folie_a_deux.main.setup_logging")
    def test_main_api_base_override(self, mock_setup_logging):
        """Test API base URL override."""
        with patch("folie_a_deux.main.folie_a_deux") as mock_exp:
            mock_result = Mock()
            mock_result.summary.return_value = {}
            mock_result.rounds = []
            mock_exp.return_value = mock_result

            args = ["--api-base", "http://custom:8080"]
            exit_code = main(args)

            assert exit_code == 0
            config = mock_exp.call_args[1]["config"]
            assert config.api_base == "http://custom:8080"


class TestMainIntegration:
    """Integration tests for main function."""

    def test_main_can_parse_all_arguments(self):
        """Test that main can parse all documented arguments without error."""
        # This test ensures the argument parser is set up correctly
        with (
            patch("folie_a_deux.main.folie_a_deux") as mock_exp,
            patch("folie_a_deux.main.setup_logging"),
        ):
            mock_result = Mock()
            mock_result.summary.return_value = {}
            mock_result.rounds = []
            mock_exp.return_value = mock_result

            # Test all possible arguments
            args = [
                "--rounds",
                "10",
                "--alpha",
                "0.25",
                "--use-cot",
                "--model",
                "test/model",
                "--api-base",
                "http://test:1234",
                "--temperature",
                "0.9",
                "--max-tokens",
                "512",
                "--log-level",
                "WARNING",
                "--quiet",
            ]

            exit_code = main(args)
            assert exit_code == 0

    def test_main_help_returns_zero(self):
        """Test that help command returns successfully."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        # Help should exit with code 0
        assert exc_info.value.code == 0

    @patch("folie_a_deux.main.folie_a_deux")
    @patch("folie_a_deux.main.setup_logging")
    def test_main_no_args_uses_defaults(self, mock_setup_logging, mock_folie_a_deux):
        """Test that main with no args uses default configuration."""
        mock_result = Mock()
        mock_result.summary.return_value = {}
        mock_result.rounds = []
        mock_folie_a_deux.return_value = mock_result

        exit_code = main([])

        assert exit_code == 0
        # Should use default log level
        mock_setup_logging.assert_called_once_with("INFO")

        # Should call experiment with config from environment
        mock_folie_a_deux.assert_called_once()
        call_kwargs = mock_folie_a_deux.call_args[1]
        assert "config" in call_kwargs
        assert call_kwargs["use_cot"] is False  # Default value
