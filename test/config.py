"""Configuration helper for tests."""

from os import getenv
from sys import exit
from typing import NoReturn

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - gracefully handle missing package
    def load_dotenv(*_args, **_kwargs):
        """Fallback load_dotenv when python-dotenv is unavailable."""

        return False

from utils.logger import logger


class Config:
    """Configuration used in unit tests."""

    data_path: str
    test_data_file: str
    model_path: str
    model_file: str
    metric_path: str
    metric_file: str

    def __init__(self) -> None:
        load_dotenv()

        if getenv("DATA_PATH") is None:
            self.exit_program("DATA_PATH")
        else:
            self.data_path = data_path  # pragma: no cover

        test_data_file = getenv("TEST_DATA_FILE")
        if test_data_file is None:
            self.exit_program("TEST_DATA_FILE")
        else:
            self.test_data_file = test_data_file  # pragma: no cover

        model_path = getenv("MODEL_PATH")
        if model_path is None:
            self.exit_program("MODEL_PATH")
        else:
            self.model_path = model_path  # pragma: no cover

        model_file = getenv("MODEL_FILE")
        if model_file is None:
            self.exit_program("MODEL_FILE")
        else:
            self.model_file = model_file  # pragma: no cover

        metric_path = getenv("METRIC_PATH")
        if metric_path is None:
            self.exit_program("METRIC_PATH")
        else:
            self.metric_path = metric_path  # pragma: no cover

        metric_file = getenv("METRIC_FILE")
        if metric_file is None:
            self.exit_program("METRIC_FILE")
        else:
            self.metric_file = metric_file  # pragma: no cover

    def exit_program(self, env_var: str) -> None:
        """Exit test execution when a variable is missing."""
        error_message = (
            f"test: {env_var} is missing from the set environment variables."
        )
        logger.error(error_message)
        exit(error_message)
