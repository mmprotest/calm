"""Top-level package for the CALM implementation."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:  # pragma: no cover - fallback for editable installs
    __version__ = version("calm-lm")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.1.0"

DEFAULT_CONFIG_PATH = "config/defaults.yaml"
