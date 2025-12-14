from __future__ import annotations
import logging
import time
from typing import Any
from teleoperators.omni.omni import OmniTeleoperator
from typing_extensions import override

from teleoperators.omni.config import OmniConfig
from teleoperators.controller import Controller
from teleoperators.twocontroller.config import TwoControllerConfig

log = logging.getLogger(__name__)


class TwoController(Controller):
    """
    Unified controller for dual Omni/Touch haptic devices.
    - One scheduler
    - Shared filters and scaling
    - Per-device callbacks at construction
    """

    def __init__(self, config: TwoControllerConfig) -> None:
        self.left = config.left(config.left_config)
        self.right = config.right(config.right_config)

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "left": self.left.action_features,
            "right": self.right.action_features,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {
            "left": self.left.feedback_features,
            "right": self.right.feedback_features,
        }

    @property
    def is_connected(self) -> bool:
        return self.left.is_connected and self.right.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.left.is_calibrated and self.right.is_calibrated

    @override
    def connect(self, calibrate: bool = True) -> None:
        log.info("Connected to omni controller")
        self.connected = True
        self.left.connect()
        self.right.connect()

    @override
    def disconnect(self) -> None:
        log.info("Disconnecting from omni controller")
        if self.connected:
            self.left.disconnect()
            self.right.disconnect()
            self.connected = False

    @override
    def reset(self) -> None:
        self.left.reset()
        self.right.reset()

    @override
    def get_action(self) -> dict[str, Any]:
        return {
            "left": self.left.get_action(),
            "right": self.right.get_action(),
        }

    @override
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        self.left.send_feedback(feedback)
        self.right.send_feedback(feedback)

    @override
    def motion_scale(self) -> float:
        self.right.set_motion_scale(self.left.motion_scale())
        return self.left.motion_scale()

    @override
    def configure(self) -> None:
        self.left.configure()
        self.right.configure()

    @override
    def calibrate(self) -> None:
        self.left.calibrate()
        self.right.calibrate()


if __name__ == "__main__":
    left_config = OmniConfig(device_name="USB")
    right_config = OmniConfig(device_name="LAN")
    twocontroller = TwoController(
        config=TwoControllerConfig(
            left=OmniTeleoperator,
            right=OmniTeleoperator,
            left_config=left_config,
            right_config=right_config,
        )
    )
    twocontroller.connect()
    try:
        while True:
            action = twocontroller.get_action()
            print(action)
            time.sleep(0.009)
    except KeyboardInterrupt:
        pass
    finally:
        twocontroller.disconnect()
