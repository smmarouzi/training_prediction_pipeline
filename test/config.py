"""Configuration helper for tests."""

from os import getenv
from sys import exit

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - gracefully handle missing package
    def load_dotenv(*_args, **_kwargs):
        """Fallback load_dotenv when python-dotenv is unavailable."""

        return False

from utils.logger import logger


class Config:
    """Load and validate required environment variables for tests."""

    def __init__(self) -> None:
        load_dotenv()

        if getenv("DATA_PATH") is None:
            self.exit_program("DATA_PATH")
        else:
            self.data_path = getenv("DATA_PATH")

        if getenv("TEST_DATA_FILE") is None:
            self.exit_program("TEST_DATA_FILE")
        else:
            self.test_data_file = getenv("TEST_DATA_FILE")

        if getenv("MODEL_PATH") is None:
            self.exit_program("MODEL_PATH")
        else:
            self.model_path = getenv("MODEL_PATH")

        if getenv("MODEL_FILE") is None:
            self.exit_program("MODEL_FILE")
        else:
            self.model_file = getenv("MODEL_FILE")

        if getenv("METRIC_PATH") is None:
            self.exit_program("METRIC_PATH")
        else:
            self.metric_path = getenv("METRIC_PATH")

        if getenv("METRIC_FILE") is None:
            self.exit_program("METRIC_FILE")
        else:
            self.metric_file = getenv("METRIC_FILE")

    def exit_program(self, env_var: str) -> None:
        """Exit test execution when a variable is missing."""

        error_message = (
            f"test: {env_var} is missing from the set environment variables."
        )
        logger.error(error_message)
        exit(error_message)
