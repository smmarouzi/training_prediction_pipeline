from os import getenv
from sys import exit
from typing import NoReturn


class Config:
    """Validate required environment variables for the serving application."""

    ml_base_uri: str

    def __init__(self) -> None:
        ml_base_uri = getenv("ML_BASE_URI")
        if ml_base_uri is None:
            self.exit_program("ML_BASE_URI")
        else:
            self.ml_base_uri = ml_base_uri  # pragma: no cover

    def exit_program(self, env_var: str) -> NoReturn:
        error_message = (
            f"SERVER: {env_var} is missing from the set of environment variables."
        )
        exit(f"{error_message}")