from dataclasses import dataclass, field
from typing import Final, NewType, TypeAlias

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

from cameraSpec import CAMERAS, CameraSpec

AngleDeg = NewType("AngleDeg", float)

JointRange: TypeAlias = tuple[AngleDeg, AngleDeg]


XARM_DEFAULT_IP: Final[str] = "192.168.1.236"
XARM_DEFAULT_PORT: Final[int] = 0

DEFAULT_CAMERA_SPECS: Final[list[CameraSpec]] = CAMERAS

DEFAULT_START_POS: Final[tuple[float, ...]] = (0, 0, 0, 0, 0, 0, 0)


@RobotConfig.register_subclass("xarm")
@dataclass(slots=True)
class XArmConfig(RobotConfig):
    ip: str = XARM_DEFAULT_IP

    port: int = XARM_DEFAULT_PORT

    num_joints: int = 7

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            spec.name: spec.config for spec in DEFAULT_CAMERA_SPECS
        }
    )

    start_pos: tuple[float, ...] = DEFAULT_START_POS

    dt: float = 1.0 / 250.0