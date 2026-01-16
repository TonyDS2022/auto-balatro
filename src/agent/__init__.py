"""Reinforcement learning agent module for Auto-Balatro.

This module provides:
- Neural network policies (ActorCritic)
- PPO training loop
- Model evaluation utilities
- Live gameplay player
"""

from .policy import (
    ActorCritic,
    BalatroNetwork,
    ActorHead,
    CriticHead,
    CardSelectionNetwork,
    create_policy,
)
from .trainer import (
    PPOTrainer,
    TrainerConfig,
    CurriculumConfig,
    RolloutBuffer,
)
from .evaluator import (
    PolicyEvaluator,
    EpisodeResult,
    EvaluationResult,
    RandomBaseline,
    evaluate_checkpoint,
    benchmark_against_random,
)
from .player import (
    LivePlayer,
    HumanAssistedPlayer,
    PlaySession,
    PlayerState,
    create_player,
)

__all__ = [
    # Policy networks
    "ActorCritic",
    "BalatroNetwork",
    "ActorHead",
    "CriticHead",
    "CardSelectionNetwork",
    "create_policy",
    # Training
    "PPOTrainer",
    "TrainerConfig",
    "CurriculumConfig",
    "RolloutBuffer",
    # Evaluation
    "PolicyEvaluator",
    "EpisodeResult",
    "EvaluationResult",
    "RandomBaseline",
    "evaluate_checkpoint",
    "benchmark_against_random",
    # Live play
    "LivePlayer",
    "HumanAssistedPlayer",
    "PlaySession",
    "PlayerState",
    "create_player",
]
