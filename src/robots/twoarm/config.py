from dataclasses import dataclass, field
from typing import Final, NewType

from lerobot.cameras import CameraConfig
from lerobot.robots import Robot, RobotConfig

from cameraSpec import CAMERAS, CameraSpec

AngleDeg = NewType("AngleDeg", float)

DEFAULT_CAMERA_SPECS: Final[list[CameraSpec]] = CAMERAS


@RobotConfig.register_subclass("twoarm")
@dataclass(slots=True)
class TwoArmConfig(RobotConfig):
    left_config: RobotConfig
    right_config: RobotConfig

    left: Robot
    right: Robot

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            spec.name: spec.config for spec in DEFAULT_CAMERA_SPECS
        }
    )
