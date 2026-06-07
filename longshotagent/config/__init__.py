"""
Configuration management for the video agent pipeline.

Provides flexible configuration loading from YAML files with CLI override support.
"""

from .config_manager import ConfigManager, load_config

__all__ = [
    "ConfigManager",
    "load_config"
]