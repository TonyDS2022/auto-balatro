"""Action executor module for Auto-Balatro.

Provides mouse control, game action execution, and coordination
for automated game interaction.
"""

from .input_backend import (
    InputEvent,
    InputBackend,
    PyAutoGUIBackend,
    MockInputBackend,
    create_backend,
)
from .mouse import (
    Point,
    Region,
    MouseController,
    SafeMouseController,
)
from .actions import (
    GameAction,
    UILayout,
    ActionResult,
    ActionExecutor,
)
from .coordinator import (
    CoordinatorState,
    ActionStep,
    ActionSequence,
    CoordinatorConfig,
    ActionCoordinator,
    GameController,
)

__all__ = [
    # Input backend
    "InputEvent",
    "InputBackend",
    "PyAutoGUIBackend",
    "MockInputBackend",
    "create_backend",
    # Mouse control
    "Point",
    "Region",
    "MouseController",
    "SafeMouseController",
    # Actions
    "GameAction",
    "UILayout",
    "ActionResult",
    "ActionExecutor",
    # Coordination
    "CoordinatorState",
    "ActionStep",
    "ActionSequence",
    "CoordinatorConfig",
    "ActionCoordinator",
    "GameController",
]
