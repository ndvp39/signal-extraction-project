"""
Shared pytest fixtures for the signal extraction test suite.

Fixtures defined here are available to all tests without explicit import.
Add shared test data, config objects, and signal bundles here as they are needed.
"""

import pytest

from signal_extraction.shared.config import ConfigManager

CONFIG_PATH = "config/setup.json"


@pytest.fixture(scope="session")
def config() -> ConfigManager:
    """Return a ConfigManager loaded from the real setup.json.

    Session-scoped so the file is read only once per test run.
    """
    return ConfigManager(CONFIG_PATH)
