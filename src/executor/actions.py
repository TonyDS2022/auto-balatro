"""Game action definitions and executor for Balatro.

Maps abstract game actions to concrete mouse interactions
based on detected UI element positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable

from src.executor.mouse import MouseController, Point, Region

if TYPE_CHECKING:
    from src.vision.state_detector import GameStateInfo

logger = logging.getLogger(__name__)


class GameAction(Enum):
    """All possible game actions."""

    # Card selection
    SELECT_CARD = auto()
    DESELECT_CARD = auto()
    TOGGLE_CARD = auto()

    # Hand actions
    PLAY_HAND = auto()
    DISCARD = auto()
    SORT_HAND = auto()

    # Blind selection
    SELECT_SMALL_BLIND = auto()
    SELECT_BIG_BLIND = auto()
    SELECT_BOSS_BLIND = auto()
    SKIP_BLIND = auto()

    # Shop actions
    BUY_ITEM = auto()
    SELL_JOKER = auto()
    REROLL_SHOP = auto()
    END_SHOP = auto()

    # Consumables
    USE_CONSUMABLE = auto()
    SELECT_CONSUMABLE = auto()

    # Joker management
    MOVE_JOKER = auto()
    SELECT_JOKER = auto()

    # General
    CLICK_BUTTON = auto()
    WAIT = auto()
    CONFIRM = auto()
    CANCEL = auto()


@dataclass
class UILayout:
    """Screen positions for UI elements.

    All positions are relative to the game window.
    These can be calibrated or detected dynamically.
    """

    # Window bounds
    window: Region = field(default_factory=lambda: Region(0, 0, 1920, 1080))

    # Card positions (8 card slots)
    card_slots: list[Region] = field(default_factory=list)

    # Action buttons
    play_button: Region | None = None
    discard_button: Region | None = None
    sort_button: Region | None = None

    # Blind selection
    small_blind_button: Region | None = None
    big_blind_button: Region | None = None
    boss_blind_button: Region | None = None
    skip_blind_button: Region | None = None

    # Shop elements
    shop_items: list[Region] = field(default_factory=list)
    reroll_button: Region | None = None
    end_shop_button: Region | None = None

    # Joker slots (5 slots typically)
    joker_slots: list[Region] = field(default_factory=list)

    # Consumable slots (2 slots typically)
    consumable_slots: list[Region] = field(default_factory=list)

    # Dialog buttons
    confirm_button: Region | None = None
    cancel_button: Region | None = None

    @classmethod
    def default_1080p(cls) -> UILayout:
        """Create default layout for 1920x1080 resolution.

        Returns:
            UILayout with standard Balatro positions.
        """
        layout = cls()
        layout.window = Region(0, 0, 1920, 1080)

        # Card positions (approximate, centered at bottom)
        card_width = 100
        card_height = 140
        card_y = 800
        card_start_x = 560  # Center 8 cards

        layout.card_slots = [
            Region(card_start_x + i * (card_width + 10), card_y, card_width, card_height)
            for i in range(8)
        ]

        # Action buttons (right side)
        layout.play_button = Region(1600, 700, 200, 60)
        layout.discard_button = Region(1600, 780, 200, 60)
        layout.sort_button = Region(1600, 620, 200, 40)

        # Blind selection (center screen)
        layout.small_blind_button = Region(400, 400, 300, 200)
        layout.big_blind_button = Region(810, 400, 300, 200)
        layout.boss_blind_button = Region(1220, 400, 300, 200)
        layout.skip_blind_button = Region(860, 650, 200, 50)

        # Shop (center area)
        shop_item_width = 150
        shop_y = 300
        layout.shop_items = [
            Region(300 + i * (shop_item_width + 20), shop_y, shop_item_width, 200)
            for i in range(5)
        ]
        layout.reroll_button = Region(860, 550, 200, 50)
        layout.end_shop_button = Region(860, 620, 200, 50)

        # Jokers (top left)
        joker_width = 80
        joker_y = 50
        layout.joker_slots = [
            Region(50 + i * (joker_width + 10), joker_y, joker_width, 110)
            for i in range(5)
        ]

        # Consumables (top right)
        layout.consumable_slots = [
            Region(1700, 50, 80, 110),
            Region(1790, 50, 80, 110),
        ]

        # Dialog buttons
        layout.confirm_button = Region(800, 600, 150, 50)
        layout.cancel_button = Region(970, 600, 150, 50)

        return layout


@dataclass
class ActionResult:
    """Result of executing an action."""

    success: bool
    action: GameAction
    message: str = ""
    data: dict = field(default_factory=dict)


class ActionExecutor:
    """Executes game actions via mouse control.

    Translates abstract game actions into concrete mouse
    movements and clicks based on UI layout.

    Attributes:
        mouse: MouseController instance.
        layout: Current UI layout configuration.
    """

    def __init__(
        self,
        mouse: MouseController | None = None,
        layout: UILayout | None = None,
    ) -> None:
        """Initialize action executor.

        Args:
            mouse: Mouse controller (created if None).
            layout: UI layout (default 1080p if None).
        """
        self.mouse = mouse or MouseController()
        self.layout = layout or UILayout.default_1080p()

        # Set mouse bounds to window
        if self.layout.window:
            self.mouse.bounds = self.layout.window

        # Action handlers
        self._handlers: dict[GameAction, Callable] = {
            GameAction.SELECT_CARD: self._select_card,
            GameAction.DESELECT_CARD: self._deselect_card,
            GameAction.TOGGLE_CARD: self._toggle_card,
            GameAction.PLAY_HAND: self._play_hand,
            GameAction.DISCARD: self._discard,
            GameAction.SORT_HAND: self._sort_hand,
            GameAction.SELECT_SMALL_BLIND: self._select_small_blind,
            GameAction.SELECT_BIG_BLIND: self._select_big_blind,
            GameAction.SELECT_BOSS_BLIND: self._select_boss_blind,
            GameAction.SKIP_BLIND: self._skip_blind,
            GameAction.BUY_ITEM: self._buy_item,
            GameAction.SELL_JOKER: self._sell_joker,
            GameAction.REROLL_SHOP: self._reroll_shop,
            GameAction.END_SHOP: self._end_shop,
            GameAction.USE_CONSUMABLE: self._use_consumable,
            GameAction.SELECT_CONSUMABLE: self._select_consumable,
            GameAction.SELECT_JOKER: self._select_joker,
            GameAction.CLICK_BUTTON: self._click_button,
            GameAction.WAIT: self._wait,
            GameAction.CONFIRM: self._confirm,
            GameAction.CANCEL: self._cancel,
        }

        logger.info("ActionExecutor initialized")

    def execute(
        self,
        action: GameAction,
        **kwargs,
    ) -> ActionResult:
        """Execute a game action.

        Args:
            action: Action to execute.
            **kwargs: Action-specific parameters.

        Returns:
            ActionResult indicating success/failure.
        """
        handler = self._handlers.get(action)
        if handler is None:
            return ActionResult(
                success=False,
                action=action,
                message=f"No handler for action: {action}",
            )

        try:
            result = handler(**kwargs)
            return result
        except Exception as e:
            logger.error(f"Action {action} failed: {e}")
            return ActionResult(
                success=False,
                action=action,
                message=str(e),
            )

    def update_layout(self, layout: UILayout) -> None:
        """Update UI layout.

        Args:
            layout: New UI layout configuration.
        """
        self.layout = layout
        if layout.window:
            self.mouse.bounds = layout.window

    def update_layout_from_state(self, state: GameStateInfo) -> None:
        """Update layout from detected game state.

        Args:
            state: Detected game state with UI positions.
        """
        # Update card positions from detected cards
        if hasattr(state, "card_positions") and state.card_positions:
            self.layout.card_slots = [
                Region(pos[0] - 50, pos[1] - 70, 100, 140)
                for pos in state.card_positions
            ]

        # Update button positions if detected
        if hasattr(state, "button_positions"):
            buttons = state.button_positions
            if "play" in buttons:
                pos = buttons["play"]
                self.layout.play_button = Region(pos[0] - 100, pos[1] - 30, 200, 60)
            if "discard" in buttons:
                pos = buttons["discard"]
                self.layout.discard_button = Region(pos[0] - 100, pos[1] - 30, 200, 60)

    # Card actions
    def _select_card(self, card_index: int = 0, **kwargs) -> ActionResult:
        """Select a card by index."""
        if card_index >= len(self.layout.card_slots):
            return ActionResult(
                success=False,
                action=GameAction.SELECT_CARD,
                message=f"Invalid card index: {card_index}",
            )

        region = self.layout.card_slots[card_index]
        self.mouse.click(region.center)

        return ActionResult(
            success=True,
            action=GameAction.SELECT_CARD,
            data={"card_index": card_index},
        )

    def _deselect_card(self, card_index: int = 0, **kwargs) -> ActionResult:
        """Deselect a card (same as select - toggle behavior)."""
        return self._select_card(card_index)

    def _toggle_card(self, card_index: int = 0, **kwargs) -> ActionResult:
        """Toggle card selection."""
        return self._select_card(card_index)

    def _play_hand(self, **kwargs) -> ActionResult:
        """Click the play hand button."""
        if self.layout.play_button is None:
            return ActionResult(
                success=False,
                action=GameAction.PLAY_HAND,
                message="Play button not configured",
            )

        self.mouse.click(self.layout.play_button.center)
        return ActionResult(success=True, action=GameAction.PLAY_HAND)

    def _discard(self, **kwargs) -> ActionResult:
        """Click the discard button."""
        if self.layout.discard_button is None:
            return ActionResult(
                success=False,
                action=GameAction.DISCARD,
                message="Discard button not configured",
            )

        self.mouse.click(self.layout.discard_button.center)
        return ActionResult(success=True, action=GameAction.DISCARD)

    def _sort_hand(self, **kwargs) -> ActionResult:
        """Click the sort button."""
        if self.layout.sort_button is None:
            return ActionResult(
                success=False,
                action=GameAction.SORT_HAND,
                message="Sort button not configured",
            )

        self.mouse.click(self.layout.sort_button.center)
        return ActionResult(success=True, action=GameAction.SORT_HAND)

    # Blind selection
    def _select_small_blind(self, **kwargs) -> ActionResult:
        """Select small blind."""
        if self.layout.small_blind_button is None:
            return ActionResult(
                success=False,
                action=GameAction.SELECT_SMALL_BLIND,
                message="Small blind button not configured",
            )

        self.mouse.click(self.layout.small_blind_button.center)
        return ActionResult(success=True, action=GameAction.SELECT_SMALL_BLIND)

    def _select_big_blind(self, **kwargs) -> ActionResult:
        """Select big blind."""
        if self.layout.big_blind_button is None:
            return ActionResult(
                success=False,
                action=GameAction.SELECT_BIG_BLIND,
                message="Big blind button not configured",
            )

        self.mouse.click(self.layout.big_blind_button.center)
        return ActionResult(success=True, action=GameAction.SELECT_BIG_BLIND)

    def _select_boss_blind(self, **kwargs) -> ActionResult:
        """Select boss blind."""
        if self.layout.boss_blind_button is None:
            return ActionResult(
                success=False,
                action=GameAction.SELECT_BOSS_BLIND,
                message="Boss blind button not configured",
            )

        self.mouse.click(self.layout.boss_blind_button.center)
        return ActionResult(success=True, action=GameAction.SELECT_BOSS_BLIND)

    def _skip_blind(self, **kwargs) -> ActionResult:
        """Skip current blind."""
        if self.layout.skip_blind_button is None:
            return ActionResult(
                success=False,
                action=GameAction.SKIP_BLIND,
                message="Skip blind button not configured",
            )

        self.mouse.click(self.layout.skip_blind_button.center)
        return ActionResult(success=True, action=GameAction.SKIP_BLIND)

    # Shop actions
    def _buy_item(self, item_index: int = 0, **kwargs) -> ActionResult:
        """Buy an item from the shop."""
        if item_index >= len(self.layout.shop_items):
            return ActionResult(
                success=False,
                action=GameAction.BUY_ITEM,
                message=f"Invalid shop item index: {item_index}",
            )

        region = self.layout.shop_items[item_index]
        self.mouse.click(region.center)

        return ActionResult(
            success=True,
            action=GameAction.BUY_ITEM,
            data={"item_index": item_index},
        )

    def _sell_joker(self, joker_index: int = 0, **kwargs) -> ActionResult:
        """Sell a joker."""
        if joker_index >= len(self.layout.joker_slots):
            return ActionResult(
                success=False,
                action=GameAction.SELL_JOKER,
                message=f"Invalid joker index: {joker_index}",
            )

        region = self.layout.joker_slots[joker_index]
        # Drag to sell area or right-click to sell
        self.mouse.right_click(region.center)

        return ActionResult(
            success=True,
            action=GameAction.SELL_JOKER,
            data={"joker_index": joker_index},
        )

    def _reroll_shop(self, **kwargs) -> ActionResult:
        """Reroll the shop."""
        if self.layout.reroll_button is None:
            return ActionResult(
                success=False,
                action=GameAction.REROLL_SHOP,
                message="Reroll button not configured",
            )

        self.mouse.click(self.layout.reroll_button.center)
        return ActionResult(success=True, action=GameAction.REROLL_SHOP)

    def _end_shop(self, **kwargs) -> ActionResult:
        """End the shop phase."""
        if self.layout.end_shop_button is None:
            return ActionResult(
                success=False,
                action=GameAction.END_SHOP,
                message="End shop button not configured",
            )

        self.mouse.click(self.layout.end_shop_button.center)
        return ActionResult(success=True, action=GameAction.END_SHOP)

    # Consumable actions
    def _use_consumable(self, slot_index: int = 0, **kwargs) -> ActionResult:
        """Use a consumable."""
        if slot_index >= len(self.layout.consumable_slots):
            return ActionResult(
                success=False,
                action=GameAction.USE_CONSUMABLE,
                message=f"Invalid consumable slot: {slot_index}",
            )

        region = self.layout.consumable_slots[slot_index]
        self.mouse.click(region.center)

        return ActionResult(
            success=True,
            action=GameAction.USE_CONSUMABLE,
            data={"slot_index": slot_index},
        )

    def _select_consumable(self, slot_index: int = 0, **kwargs) -> ActionResult:
        """Select a consumable for viewing."""
        return self._use_consumable(slot_index)

    def _select_joker(self, joker_index: int = 0, **kwargs) -> ActionResult:
        """Select a joker."""
        if joker_index >= len(self.layout.joker_slots):
            return ActionResult(
                success=False,
                action=GameAction.SELECT_JOKER,
                message=f"Invalid joker index: {joker_index}",
            )

        region = self.layout.joker_slots[joker_index]
        self.mouse.click(region.center)

        return ActionResult(
            success=True,
            action=GameAction.SELECT_JOKER,
            data={"joker_index": joker_index},
        )

    # General actions
    def _click_button(self, region: Region | None = None, **kwargs) -> ActionResult:
        """Click a specific button region."""
        if region is None:
            return ActionResult(
                success=False,
                action=GameAction.CLICK_BUTTON,
                message="No region specified",
            )

        self.mouse.click(region.center)
        return ActionResult(success=True, action=GameAction.CLICK_BUTTON)

    def _wait(self, duration: float = 0.5, **kwargs) -> ActionResult:
        """Wait for a duration."""
        self.mouse.wait(duration)
        return ActionResult(
            success=True,
            action=GameAction.WAIT,
            data={"duration": duration},
        )

    def _confirm(self, **kwargs) -> ActionResult:
        """Click confirm button."""
        if self.layout.confirm_button is None:
            return ActionResult(
                success=False,
                action=GameAction.CONFIRM,
                message="Confirm button not configured",
            )

        self.mouse.click(self.layout.confirm_button.center)
        return ActionResult(success=True, action=GameAction.CONFIRM)

    def _cancel(self, **kwargs) -> ActionResult:
        """Click cancel button."""
        if self.layout.cancel_button is None:
            return ActionResult(
                success=False,
                action=GameAction.CANCEL,
                message="Cancel button not configured",
            )

        self.mouse.click(self.layout.cancel_button.center)
        return ActionResult(success=True, action=GameAction.CANCEL)

    # Convenience methods for common action sequences
    def select_cards(self, indices: list[int]) -> list[ActionResult]:
        """Select multiple cards by indices.

        Args:
            indices: List of card indices to select.

        Returns:
            List of ActionResults for each selection.
        """
        results = []
        for idx in indices:
            result = self.execute(GameAction.SELECT_CARD, card_index=idx)
            results.append(result)
            self.mouse.wait(0.1)  # Small delay between selections
        return results

    def play_selected(self) -> ActionResult:
        """Play the currently selected cards.

        Returns:
            ActionResult for the play action.
        """
        return self.execute(GameAction.PLAY_HAND)

    def discard_selected(self) -> ActionResult:
        """Discard the currently selected cards.

        Returns:
            ActionResult for the discard action.
        """
        return self.execute(GameAction.DISCARD)
