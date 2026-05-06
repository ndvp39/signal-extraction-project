"""
Configuration manager for the signal extraction project.

Loads and validates config/setup.json. All configurable values (frequencies,
amplitudes, hyperparameters, paths) must be read through this module —
never hard-coded in source files.
"""

import json
from pathlib import Path
from typing import Any

from signal_extraction.constants import EXPECTED_CONFIG_VERSION


class ConfigError(Exception):
    """Raised when the configuration file is missing, malformed, or incompatible."""


class ConfigManager:
    """
    Loads setup.json and provides typed accessors for all configuration sections.

    Why a class rather than a module-level dict: encapsulation lets us validate
    once on construction and raise early, rather than crashing mid-experiment.
    """

    def __init__(self, config_path: str | Path) -> None:
        """
        Load and validate the configuration file.

        Args:
            config_path: Path to setup.json (absolute or relative to cwd).

        Raises:
            ConfigError: If the file is missing, not valid JSON, or has
                         an incompatible version field.
        """
        self._path = Path(config_path)
        self._data: dict[str, Any] = self._load()
        self._validate_version()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        """Read and parse the JSON config file."""
        if not self._path.exists():
            raise ConfigError(f"Config file not found: {self._path}")
        try:
            with self._path.open(encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Config file is not valid JSON: {exc}") from exc

    def _validate_version(self) -> None:
        """Ensure the config version is compatible with this codebase version."""
        version = self._data.get("version")
        if version != EXPECTED_CONFIG_VERSION:
            raise ConfigError(
                f"Config version mismatch: expected {EXPECTED_CONFIG_VERSION!r}, "
                f"got {version!r}. Update config/setup.json."
            )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Retrieve a nested value using a sequence of keys.

        Example:
            cfg.get("training", "learning_rate")  # returns 0.001

        Args:
            *keys:   Sequence of keys forming a path into the config dict.
            default: Value returned when the path does not exist.

        Returns:
            The value at the given path, or default if not found.
        """
        node: Any = self._data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def require(self, *keys: str) -> Any:
        """
        Like get() but raises ConfigError if the key path is missing.

        Use this for values that have no sensible default and must be present.
        """
        sentinel = object()
        value = self.get(*keys, default=sentinel)
        if value is sentinel:
            raise ConfigError(f"Required config key missing: {' -> '.join(keys)}")
        return value

    @property
    def signals(self) -> dict[str, Any]:
        """Return the full 'signals' section of the config."""
        return self.require("signals")

    @property
    def dataset(self) -> dict[str, Any]:
        """Return the full 'dataset' section of the config."""
        return self.require("dataset")

    @property
    def training(self) -> dict[str, Any]:
        """Return the full 'training' section of the config."""
        return self.require("training")

    @property
    def models(self) -> dict[str, Any]:
        """Return the full 'models' section of the config."""
        return self.require("models")

    @property
    def paths(self) -> dict[str, Any]:
        """Return the full 'paths' section of the config."""
        return self.require("paths")
