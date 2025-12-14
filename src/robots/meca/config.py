from dataclasses import dataclass, field
from typing import Final, NewType, TypeAlias

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

from cameraSpec import CAMERAS, CameraSpec

AngleDeg = NewType("AngleDeg", float)

JointRange: TypeAlias = tuple[AngleDeg, AngleDeg]


MECA_DEFAULT_IP: Final[str] = "192.168.0.100"
MECA_DEFAULT_PORT: Final[int] = 3000
MECA_DEFAULT_JOINT_RANGES: Final[tuple[JointRange, ...]] = (
    (AngleDeg(-175), AngleDeg(175)),
    (AngleDeg(-70), AngleDeg(90)),
    (AngleDeg(-135), AngleDeg(70)),
    (AngleDeg(-170), AngleDeg(170)),
    (AngleDeg(-115), AngleDeg(115)),
    (AngleDeg(-40000), AngleDeg(40000)),
)

DEFAULT_CAMERA_SPECS: Final[list[CameraSpec]] = CAMERAS


@RobotConfig.register_subclass("meca")
@dataclass(slots=True)
class MecaConfig(RobotConfig):
    ip: str = MECA_DEFAULT_IP

    port: int = MECA_DEFAULT_PORT

    joint_ranges: tuple[JointRange, ...] = field(
        default_factory=lambda: tuple(MECA_DEFAULT_JOINT_RANGES)
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            spec.name: spec.config for spec in DEFAULT_CAMERA_SPECS
        }
    )

    start_pos: tuple[float, ...] | None = None
