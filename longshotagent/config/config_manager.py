"""
Configuration management system for the video agent pipeline.

Supports loading configuration from YAML files with CLI argument overrides
and environment variable substitution.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PreprocessingConfig(BaseModel):
    """Configuration for video preprocessing pipeline."""

    whisper_model: str = Field(
        default="small",
        description="Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)",
    )
    siglip_model: str = Field(
        default="ViT-B-16-SigLIP-512", description="SigLIP model name"
    )
    cache_dir: str = Field(default="./cache", description="Cache directory path")
    db_path: str = Field(default="./chroma_db", description="ChromaDB database path")
    device: str = Field(
        default="auto", description="Device for processing (cpu/cuda/auto)"
    )


class AgentConfig(BaseModel):
    """Configuration for the video agent."""

    # Main LLM server settings
    vllm_base_url: str = Field(
        default="http://localhost:8010/v1", description="vLLM server base URL"
    )
    model_name: str = Field(
        default="google/gemma-4-31B-it", description="LLM model name"
    )

    # VLM server settings (for visual refinement)
    vlm_base_url: str = Field(
        default="http://localhost:8011/v1", description="VLM server base URL"
    )
    vlm_model_name: str = Field(
        default="google/gemma-4-31B-it", description="VLM model name"
    )

    # ALM server settings (for audio refinement)
    alm_base_url: str = Field(
        default="http://localhost:8013/v1", description="ALM server base URL"
    )
    alm_model_name: str = Field(
        default="nvidia/audio-flamingo-3-hf", description="Audio language model name"
    )

    # Embedding server settings (served via vLLM --runner pooling)
    text_embedding_url: str = Field(
        default="http://localhost:8014/v1",
        description="Text embedding server URL (all-MiniLM-L6-v2)",
    )
    visual_embedding_url: str = Field(
        default="http://localhost:8018/v1",
        description="Visual embedding server URL (SigLIP)",
    )

    # Storage settings
    db_path: str = Field(default="./chroma_db", description="ChromaDB database path")
    videos_dir: str = Field(
        default="./videos", description="Directory containing video files"
    )
    video_search_paths: list = Field(
        default=["./videos"],
        description="List of directories to search for video files when original path not found",
    )
    system_prompt_path: str = Field(
        default="prompts/system_prompt.txt", description="System prompt file path"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )
    file_path: Optional[str] = Field(
        default="video_agent.log", description="Log file path"
    )
    console_output: bool = Field(default=True, description="Enable console logging")


class AppConfig(BaseModel):
    """Main application configuration."""

    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class ConfigManager:
    """
    Manages configuration loading and merging from multiple sources.

    Supports loading from:
    1. Default values (lowest priority)
    2. YAML configuration files
    3. Environment variables
    4. CLI arguments (highest priority)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self._config: Optional[AppConfig] = None

    def load_config(self, cli_overrides: Optional[Dict[str, Any]] = None) -> AppConfig:
        """
        Load configuration from all sources with proper precedence.

        Args:
            cli_overrides: Dictionary of CLI argument overrides

        Returns:
            Merged configuration object
        """
        # Start with default configuration
        config_dict = AppConfig().model_dump()

        # Load from file if specified
        if self.config_path and Path(self.config_path).exists():
            file_config = self._load_from_file(self.config_path)
            config_dict = self._deep_merge(config_dict, file_config)
            logger.info(f"Loaded configuration from: {self.config_path}")

        # Apply environment variable overrides
        env_config = self._load_from_env()
        config_dict = self._deep_merge(config_dict, env_config)

        # Apply CLI overrides (highest priority)
        if cli_overrides:
            config_dict = self._deep_merge(config_dict, cli_overrides)
            logger.info("Applied CLI argument overrides")

        # Create and cache the configuration
        self._config = AppConfig(**config_dict)
        return self._config

    def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Substitute environment variables
            content = self._substitute_env_vars(content)

            # Parse YAML
            config = yaml.safe_load(content)
            return config or {}

        except Exception as e:
            logger.error(f"Error loading config file {file_path}: {e}")
            return {}

    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}

        # Define environment variable mappings
        env_mappings = {
            # Preprocessing
            "VIDEO_AGENT_WHISPER_MODEL": ["preprocessing", "whisper_model"],
            "VIDEO_AGENT_SIGLIP_MODEL": ["preprocessing", "siglip_model"],
            "VIDEO_AGENT_CACHE_DIR": ["preprocessing", "cache_dir"],
            "VIDEO_AGENT_DB_PATH": ["preprocessing", "db_path"],
            "VIDEO_AGENT_DEVICE": ["preprocessing", "device"],
            # Agent
            "VIDEO_AGENT_VLLM_URL": ["agent", "vllm_base_url"],
            "VIDEO_AGENT_MODEL_NAME": ["agent", "model_name"],
            "VIDEO_AGENT_AGENT_DB_PATH": ["agent", "db_path"],
            # Logging
            "VIDEO_AGENT_LOG_LEVEL": ["logging", "level"],
            "VIDEO_AGENT_LOG_FILE": ["logging", "file_path"],
        }

        for env_var, path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(env_config, path, value)

        return env_config

    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in the format ${VAR_NAME}."""
        import re

        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.getenv(var_name, default_value)

        # Pattern: ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"
        return re.sub(pattern, replace_env_var, content)

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _set_nested_value(self, config: Dict[str, Any], path: list, value: str):
        """Set a nested configuration value using a path list."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Convert string values to appropriate types
        final_key = path[-1]
        if value.lower() in ["true", "false"]:
            current[final_key] = value.lower() == "true"
        elif value.isdigit():
            current[final_key] = int(value)
        elif self._is_float(value):
            current[final_key] = float(value)
        else:
            current[final_key] = value

    def _is_float(self, value: str) -> bool:
        """Check if a string represents a float."""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def get_config(self) -> AppConfig:
        """Get the current configuration."""
        if self._config is None:
            return self.load_config()
        return self._config

    def save_config(self, file_path: str) -> None:
        """Save current configuration to a YAML file."""
        if self._config is None:
            raise ValueError("No configuration loaded")

        config_dict = self._config.model_dump()

        with open(file_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to: {file_path}")


def load_config(
    config_path: Optional[str] = None, cli_overrides: Optional[Dict[str, Any]] = None
) -> AppConfig:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to the configuration file
        cli_overrides: Dictionary of CLI argument overrides

    Returns:
        Application configuration
    """
    manager = ConfigManager(config_path)
    return manager.load_config(cli_overrides)
