from pynput import keyboard as pkb
from typing import Any
from typing_extensions import override
from teleoperators.controller import Controller


class KeyboardController(Controller):

    def __init__(self, scale: float = 1):
        self._scale = scale
        self._gripper_state = False

        self._pressed_keys = set()
        self._listener = pkb.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.start()

    @property
    @override
    def action_features(self) -> dict:
        return {
            "dx": float,
            "dy": float,
            "dz": float,
            "da": float,
            "db": float,
            "dg": float,
            "gripper": bool,
        }

    @property
    @override
    def feedback_features(self) -> dict:
        return {}

    @property
    @override
    def is_connected(self) -> bool:
        return True

    @property
    @override
    def is_calibrated(self) -> bool:
        return True

    @override
    def connect(self, calibrate: bool = True) -> None:
        pass

    @override
    def calibrate(self) -> None:
        pass

    @override
    def configure(self) -> None:
        pass

    @override
    def get_action(self) -> dict[str, Any]:
        scale = self._scale
        dx = dy = dz = da = db = dg = 0.0

        # Translation
        if pkb.KeyCode.from_char("w") in self._pressed_keys:
            dy += scale
        if pkb.KeyCode.from_char("s") in self._pressed_keys:
            dy -= scale
        if pkb.KeyCode.from_char("a") in self._pressed_keys:
            dx -= scale
        if pkb.KeyCode.from_char("d") in self._pressed_keys:
            dx += scale
        if pkb.KeyCode.from_char("q") in self._pressed_keys:
            dz += scale
        if pkb.KeyCode.from_char("e") in self._pressed_keys:
            dz -= scale

        # Rotation
        if pkb.KeyCode.from_char("j") in self._pressed_keys:
            da += scale
        if pkb.KeyCode.from_char("l") in self._pressed_keys:
            da -= scale
        if pkb.KeyCode.from_char("i") in self._pressed_keys:
            db += scale
        if pkb.KeyCode.from_char("k") in self._pressed_keys:
            db -= scale
        if pkb.KeyCode.from_char("u") in self._pressed_keys:
            dg += scale
        if pkb.KeyCode.from_char("o") in self._pressed_keys:
            dg -= scale

        # Gripper toggle (space)
        if pkb.Key.space in self._pressed_keys:
            self._gripper_state = not self._gripper_state
            self._pressed_keys.discard(pkb.Key.space)

        return {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "da": da,
            "db": db,
            "dg": dg,
            "gripper": self._gripper_state,
        }

    @override
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @override
    def disconnect(self) -> None:
        pass

    @override
    def motion_scale(self) -> float:
        return self._scale

    def _on_press(self, key):
        self._pressed_keys.add(key)

    def _on_release(self, key):
        self._pressed_keys.discard(key)
