from dataclasses import dataclass, field
from typing import Final, NewType, TypeAlias

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

from cameraSpec import CAMERAS, CameraSpec

AngleDeg = NewType("AngleDeg", float)

JointRange: TypeAlias = tuple[AngleDeg, AngleDeg]


FAIRNO_DEFAULT_IP: Final[str] = "192.168.58.2"
FAIRNO_DEFAULT_PORT: Final[int] = 0
FAIRNO_DEFAULT_JOINT_RANGES: Final[tuple[JointRange, ...]] = (
    (AngleDeg(-175), AngleDeg(175)),
    (AngleDeg(-265), AngleDeg(85)),
    (AngleDeg(-150), AngleDeg(150)),
    (AngleDeg(-265), AngleDeg(85)),
    (AngleDeg(-175), AngleDeg(175)),
    (AngleDeg(-175), AngleDeg(175)),
)

DEFAULT_CAMERA_SPECS: Final[list[CameraSpec]] = CAMERAS

DEFAULT_START_POS: Final[tuple[float, ...]] = (
    62.082,
    -126.262,
    -66.842,
    -138.543,
    93.185,
    91.954,
    0,
)


@RobotConfig.register_subclass("fairino")
@dataclass(slots=True)
class FairinoConfig(RobotConfig):
    ip: str = FAIRNO_DEFAULT_IP

    port: int = FAIRNO_DEFAULT_PORT

    joint_ranges: tuple[JointRange, ...] = field(
        default_factory=lambda: tuple(FAIRNO_DEFAULT_JOINT_RANGES)
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            spec.name: spec.config for spec in DEFAULT_CAMERA_SPECS
        }
    )

    start_pos: tuple[float, ...] = DEFAULT_START_POS
