from dataclasses import dataclass, field
from typing import Final

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

from robots.xarm.config import XArmConfig
from cameraSpec import CAMERAS, CameraSpec

import numpy as np


# Lite6-specific defaults
LITE6_DEFAULT_IP: Final[str] = "192.168.1.166"


# Recommended safe start position (避免奇异点)
LITE6_SAFE_START_POS: Final[tuple[float, ...]] = (5.5, 29, 24, 7.5, -70.4, -3.1)

# Tool configuration
LITE6_TOOL_AXIS: Final[str] = "z"  # Tool extends along Z axis

# URDF paths
LITE6_URDF_PATH: Final[str] = "src/robots/xarm/lite6/lite6.urdf"
LITE6_PACKAGE_DIR: Final[str] = "src/robots/xarm/lite6"


# Default camera configuration
LITE6_DEFAULT_CAMERA_SPECS: Final[list[CameraSpec]] = CAMERAS


def rearrange_lite6_deltas(delta: dict[str, float]) -> np.ndarray:
    """Rearrange deltas for Lite6 configuration."""
    return np.array(
        [
            -delta.get("dz"),
            -delta.get("dx"),
            delta.get("dy"),
            -delta.get("db"),
            -delta.get("da"),
            delta.get("dg"),
        ]
    )


@RobotConfig.register_subclass("lite6")
@dataclass(slots=True)
class Lite6Config(XArmConfig):
    """Configuration for UFactory Lite6 robot arm.

    The Lite6 is a 6-DOF collaborative robot arm with:
    - Reach: 508mm
    - Payload: 3kg
    - Repeatability: ±0.05mm
    - Weight: ~10kg
    """

    # Network configuration
    ip: str = LITE6_DEFAULT_IP

    # Robot specifications
    num_joints: int = 6  # Lite6 has 6 joints (vs 7 for xArm7)

    # Camera configuration
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            spec.name: spec.config for spec in LITE6_DEFAULT_CAMERA_SPECS
        }
    )

    # Start position (joint angles in degrees)
    start_pos: tuple[float, ...] = LITE6_SAFE_START_POS

    # Kinematics configuration
    urdf_path: str = LITE6_URDF_PATH
    package_dir: str = LITE6_PACKAGE_DIR
    tool_axis: str = LITE6_TOOL_AXIS
    tool_length: float = 0.165

    def init(self) -> None:
        super().init()

    def rearrange_deltas(self, delta: dict[str, float]) -> tuple[float, ...]:
        return rearrange_lite6_deltas(delta)
