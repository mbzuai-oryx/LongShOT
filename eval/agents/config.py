"""Agent configuration loading utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

AGENTS_DIR = Path(__file__).parent


def load_agent_config(agent_name: str, override_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration for an agent.

    Args:
        agent_name: Name of the agent (e.g., "video_explorer")
        override_path: Optional path to override config file

    Returns:
        Merged configuration dictionary
    """
    config: Dict[str, Any] = {}

    agent_config_path = AGENTS_DIR / agent_name / "config.yaml"
    if agent_config_path.exists():
        with open(agent_config_path) as f:
            config = yaml.safe_load(f) or {}

    if override_path and os.path.exists(override_path):
        with open(override_path) as f:
            override = yaml.safe_load(f) or {}
            config = _deep_merge(config, override)

    env_overrides = _get_env_overrides(agent_name)
    config = _deep_merge(config, env_overrides)

    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _get_env_overrides(agent_name: str) -> Dict[str, Any]:
    """Get config overrides from environment variables.

    Environment variables should be prefixed with AGENT_{NAME}_.
    Example: AGENT_VIDEO_EXPLORER_MAX_ROUNDS=10
    """
    prefix = f"AGENT_{agent_name.upper()}_"
    overrides: Dict[str, Any] = {}

    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            try:
                overrides[config_key] = int(value)
            except ValueError:
                try:
                    overrides[config_key] = float(value)
                except ValueError:
                    if value.lower() in ("true", "false"):
                        overrides[config_key] = value.lower() == "true"
                    else:
                        overrides[config_key] = value

    return overrides


def get_repos_dir() -> Path:
    """Get the directory where agent repos are cloned."""
    return AGENTS_DIR / "repos"
