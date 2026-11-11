"""Shared test configuration."""
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_output_dir(tmp_path_factory):
    """Send everything the code under test generates to a temp folder instead of the checkout."""
    previous = os.environ.get("RCWE_OUTPUT_DIR")
    os.environ["RCWE_OUTPUT_DIR"] = str(tmp_path_factory.mktemp("rcwe_output"))
    yield
    if previous is None:
        os.environ.pop("RCWE_OUTPUT_DIR", None)
    else:
        os.environ["RCWE_OUTPUT_DIR"] = previous
