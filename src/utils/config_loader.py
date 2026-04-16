"""
Config Loader Utility

Loads YAML configuration files and provides easy access to settings.
This centralizes configuration so you don't hardcode paths/values in code.
"""

import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Simple config container with dict-like and attribute access."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __getattr__(self, key: str) -> Any:
        try:
            value = self._data[key]
            # Nested dicts become Config objects for dot access
            if isinstance(value, dict):
                return Config(value)
            return value
        except KeyError:
            raise AttributeError(f"Config has no key: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default fallback."""
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Return raw dictionary."""
        return self._data


def load_config(config_path: str = "config/config.yaml") -> Config:
    """
    Load YAML config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config object with dot notation access

    Example:
        config = load_config()
        print(config.detection.confidence_threshold)  # 0.5
    """
    # Handle relative paths from project root
    if not Path(config_path).is_absolute():
        # Try to find config relative to project root
        project_root = Path(__file__).parent.parent.parent
        full_path = project_root / config_path
    else:
        full_path = Path(config_path)

    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_path}")

    with open(full_path, 'r') as f:
        data = yaml.safe_load(f)

    return Config(data)


# Global config instance (lazy loaded)
_config_instance = None


def get_config() -> Config:
    """Get cached config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance
