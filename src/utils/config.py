"""Configuration loader for Auto-Balatro.

This module provides typed configuration access with YAML loading,
validation, environment variable overrides, and singleton access.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class CaptureConfig:
    """Screen capture configuration."""

    target_fps: int = 10
    window_title: str = "Balatro"
    use_dxcam: bool = False


@dataclass
class VisionConfig:
    """Computer vision configuration."""

    template_dir: str = "data/templates"
    confidence_threshold: float = 0.8
    grayscale: bool = True
    scale_factors: List[float] = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2])


@dataclass
class Resolution:
    """Screen resolution configuration."""

    width: int = 1920
    height: int = 1080


@dataclass
class GameConfig:
    """Game-specific configuration."""

    base_resolution: Resolution = field(default_factory=Resolution)
    max_hand_size: int = 8
    max_selection: int = 5


@dataclass
class TrainingConfig:
    """RL training hyperparameters."""

    total_timesteps: int = 1000000
    learning_rate: float = 0.0003
    batch_size: int = 64
    n_steps: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""

    enabled: bool = True
    start_ante: int = 1
    max_ante: int = 8


@dataclass
class AgentConfig:
    """RL agent configuration."""

    model_path: str = "models/balatro_ppo.zip"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


@dataclass
class ExecutorConfig:
    """Action executor configuration."""

    click_delay: float = 0.1
    animation_wait: float = 0.5
    failsafe: bool = True
    verify_actions: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    file: str = "logs/auto_balatro.log"


@dataclass
class Config:
    """Main configuration container for Auto-Balatro.

    Provides typed access to all configuration sections loaded from YAML.

    Attributes:
        capture: Screen capture settings.
        vision: Computer vision settings.
        game: Game-specific settings.
        agent: RL agent and training settings.
        executor: Action execution settings.
        logging: Logging settings.
    """

    capture: CaptureConfig = field(default_factory=CaptureConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    game: GameConfig = field(default_factory=GameConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# Global singleton instance
_config_instance: Optional[Config] = None


def _parse_resolution(data: dict) -> Resolution:
    """Parse resolution from dict."""
    return Resolution(
        width=data.get("width", 1920),
        height=data.get("height", 1080),
    )


def _parse_capture_config(data: dict) -> CaptureConfig:
    """Parse capture configuration section."""
    return CaptureConfig(
        target_fps=data.get("target_fps", 10),
        window_title=data.get("window_title", "Balatro"),
        use_dxcam=data.get("use_dxcam", False),
    )


def _parse_vision_config(data: dict) -> VisionConfig:
    """Parse vision configuration section."""
    return VisionConfig(
        template_dir=data.get("template_dir", "data/templates"),
        confidence_threshold=data.get("confidence_threshold", 0.8),
        grayscale=data.get("grayscale", True),
        scale_factors=data.get("scale_factors", [0.8, 0.9, 1.0, 1.1, 1.2]),
    )


def _parse_game_config(data: dict) -> GameConfig:
    """Parse game configuration section."""
    base_res_data = data.get("base_resolution", {})
    return GameConfig(
        base_resolution=_parse_resolution(base_res_data),
        max_hand_size=data.get("max_hand_size", 8),
        max_selection=data.get("max_selection", 5),
    )


def _parse_training_config(data: dict) -> TrainingConfig:
    """Parse training configuration section."""
    return TrainingConfig(
        total_timesteps=data.get("total_timesteps", 1000000),
        learning_rate=data.get("learning_rate", 0.0003),
        batch_size=data.get("batch_size", 64),
        n_steps=data.get("n_steps", 2048),
        n_epochs=data.get("n_epochs", 10),
        gamma=data.get("gamma", 0.99),
        gae_lambda=data.get("gae_lambda", 0.95),
        clip_range=data.get("clip_range", 0.2),
        ent_coef=data.get("ent_coef", 0.01),
    )


def _parse_curriculum_config(data: dict) -> CurriculumConfig:
    """Parse curriculum configuration section."""
    return CurriculumConfig(
        enabled=data.get("enabled", True),
        start_ante=data.get("start_ante", 1),
        max_ante=data.get("max_ante", 8),
    )


def _parse_agent_config(data: dict) -> AgentConfig:
    """Parse agent configuration section."""
    training_data = data.get("training", {})
    curriculum_data = data.get("curriculum", {})
    return AgentConfig(
        model_path=data.get("model_path", "models/balatro_ppo.zip"),
        training=_parse_training_config(training_data),
        curriculum=_parse_curriculum_config(curriculum_data),
    )


def _parse_executor_config(data: dict) -> ExecutorConfig:
    """Parse executor configuration section."""
    return ExecutorConfig(
        click_delay=data.get("click_delay", 0.1),
        animation_wait=data.get("animation_wait", 0.5),
        failsafe=data.get("failsafe", True),
        verify_actions=data.get("verify_actions", True),
    )


def _parse_logging_config(data: dict) -> LoggingConfig:
    """Parse logging configuration section."""
    return LoggingConfig(
        level=data.get("level", "INFO"),
        file=data.get("file", "logs/auto_balatro.log"),
    )


def _validate_config(data: dict) -> None:
    """Validate that required configuration sections exist."""
    if "capture" in data and not isinstance(data["capture"], dict):
        raise ValueError("'capture' section must be a dictionary")

    if "vision" in data and not isinstance(data["vision"], dict):
        raise ValueError("'vision' section must be a dictionary")

    if "game" in data and not isinstance(data["game"], dict):
        raise ValueError("'game' section must be a dictionary")

    if "agent" in data and not isinstance(data["agent"], dict):
        raise ValueError("'agent' section must be a dictionary")

    if "executor" in data and not isinstance(data["executor"], dict):
        raise ValueError("'executor' section must be a dictionary")

    if "logging" in data and not isinstance(data["logging"], dict):
        raise ValueError("'logging' section must be a dictionary")

    # Validate specific fields if present
    if "vision" in data:
        threshold = data["vision"].get("confidence_threshold")
        if threshold is not None and not (0.0 <= threshold <= 1.0):
            raise ValueError("vision.confidence_threshold must be between 0.0 and 1.0")

    if "agent" in data and "training" in data["agent"]:
        training = data["agent"]["training"]
        lr = training.get("learning_rate")
        if lr is not None and lr <= 0:
            raise ValueError("agent.training.learning_rate must be positive")

        gamma = training.get("gamma")
        if gamma is not None and not (0.0 <= gamma <= 1.0):
            raise ValueError("agent.training.gamma must be between 0.0 and 1.0")


def load_config(path: Optional[str] = None) -> Config:
    """Load and validate configuration from YAML file.

    Args:
        path: Optional path to configuration file. If None, uses
            BALATRO_CONFIG_PATH env var or defaults to 'config.yaml'.

    Returns:
        Validated Config dataclass instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration is invalid.
        yaml.YAMLError: If the YAML is malformed.
    """
    global _config_instance

    # Determine config path
    if path is None:
        path = os.environ.get("BALATRO_CONFIG_PATH", "config.yaml")

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Handle empty config file
    if data is None:
        data = {}

    # Validate
    _validate_config(data)

    # Parse sections
    config = Config(
        capture=_parse_capture_config(data.get("capture", {})),
        vision=_parse_vision_config(data.get("vision", {})),
        game=_parse_game_config(data.get("game", {})),
        agent=_parse_agent_config(data.get("agent", {})),
        executor=_parse_executor_config(data.get("executor", {})),
        logging=_parse_logging_config(data.get("logging", {})),
    )

    # Store as singleton
    _config_instance = config

    return config


def get_config() -> Config:
    """Get the global configuration singleton.

    Returns the previously loaded configuration, or loads it from
    the default path if not yet loaded.

    Returns:
        The global Config instance.
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = load_config()

    return _config_instance


def reset_config() -> None:
    """Reset the global configuration singleton.

    Useful for testing or reloading configuration.
    """
    global _config_instance
    _config_instance = None
