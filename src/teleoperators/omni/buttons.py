import logging
from dataclasses import dataclass
from enum import Enum, IntFlag
from typing import Tuple

log = logging.getLogger(__name__)


class GripperState(Enum):
    OPEN = 0
    CLOSED = 1

    def toggle(self) -> "GripperState":
        return GripperState.CLOSED if self == GripperState.OPEN else GripperState.OPEN

    @property
    def as_bool(self) -> bool:
        return self == GripperState.CLOSED


class Button(IntFlag):
    NONE = 0
    ONE = 1 << 0
    TWO = 1 << 1


@dataclass(frozen=True)
class ButtonUpdate:
    raw: int
    prev_raw: int
    rising: Button
    falling: Button
    gripper_state: GripperState
    motion_scale: float


class OmniButtons:
    """
    Encapsulates all button/edge-detection logic and the side-effects
    they cause (gripper toggle, motion-scale cycling).
    """

    def __init__(
        self,
        *,
        scales: Tuple[float, ...],
        initial_gripper: GripperState = GripperState.OPEN,
        mask: Button = Button.ONE | Button.TWO,
        buttons: Tuple[str, ...] = ("GRIPPER", "SCALE"),
    ) -> None:
        assert len(scales) >= 1, "Must provide at least one scale"
        assert len(buttons) <= 2, "Can only configure up to 2 buttons"
        assert all(
            btn in ("GRIPPER", "SCALE", "KILL") for btn in buttons
        ), "Button actions must be one of: GRIPPER, SCALE, KILL"

        self._prev_raw: int = 0
        self._gripper: GripperState = initial_gripper
        self._scales: Tuple[float, ...] = scales
        self._scale_idx: int = len(scales) - 1
        self._mask: int = int(mask)
        self._buttons: Tuple[str, ...] = buttons
        # Map button index to Button enum
        self._button_enums = [Button.ONE, Button.TWO]

    @property
    def gripper_state(self) -> GripperState:
        return self._gripper

    @property
    def motion_scale(self) -> float:
        return self._scales[self._scale_idx]

    def update(self, raw: int) -> ButtonUpdate:
        raw_m = raw & self._mask
        prev_m = self._prev_raw & self._mask

        rising_bits = (~prev_m) & raw_m
        falling_bits = prev_m & (~raw_m)

        rising = Button(rising_bits)
        falling = Button(falling_bits)

        # Handle each configured button
        for idx, action in enumerate(self._buttons):
            button_enum = self._button_enums[idx]
            if button_enum in rising:
                self._handle_button_action(action)

        self._prev_raw = raw

        return ButtonUpdate(
            raw=raw,
            prev_raw=prev_m,
            rising=rising,
            falling=falling,
            gripper_state=self._gripper,
            motion_scale=self.motion_scale,
        )
    
    def set_motion_scale(self, scale: float) -> None:
        if scale in self._scales:
            self._scale_idx = self._scales.index(scale)
        else:
            log.warning(f"Invalid motion scale: {scale}")

    def _handle_button_action(self, action: str) -> None:
        """Handle button action based on configured action type."""
        if action == "GRIPPER":
            self._gripper = self._gripper.toggle()
            log.debug("Gripper toggled -> %s", self._gripper.name)
        elif action == "SCALE":
            self._scale_idx = (self._scale_idx + 1) % len(self._scales)
            log.debug("Motion scale cycled -> %s", self.motion_scale)
        elif action == "KILL":
            log.warning(f"KILL button pressed - exiting teleop")
            exit()
        else:
            log.warning(f"Unknown button action: {action}")
