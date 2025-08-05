"""Configuration utilities for the prediction server."""

from os import getenv
from sys import exit
from typing import NoReturn

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - gracefully handle missing package
    def load_dotenv(*_args, **_kwargs):
        """Fallback load_dotenv when python-dotenv is unavailable."""

        return False


class Config:
    """Validate and load required environment variables.

    The class ensures all necessary environment variables are present and
    accessible. It will exit the application with an informative error message
    if any required variable is missing.
    """

    def __init__(self) -> None:
        load_dotenv()

        if getenv("ML_BASE_URI") is None:
            self.exit_program("ML_BASE_URI")
        else:
            self.ml_base_uri = ml_base_uri  # pragma: no cover

    def exit_program(self, env_var: str) -> None:
        """Exit the program with a helpful error message."""

        error_message = (
            f"SERVER: {env_var} is missing from the set of environment variables."
        )
        exit(error_message)

