from dataclasses import dataclass, field
from typing import Final

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

from robots.xarm.config import XArmConfig
from cameraSpec import CAMERAS, CameraSpec

import numpy as np


# xArm7-specific defaults
XARM7_DEFAULT_IP: Final[str] = "192.168.1.236"

# Recommended safe start position for xArm7
XARM7_SAFE_START_POS: Final[tuple[float, ...]] = (1.7, -8.7, 17, 26, 103, 37, 27)

# Tool configuration
XARM7_TOOL_AXIS: Final[str] = "z"  # Tool extends along Z axis

# URDF paths
XARM7_URDF_PATH: Final[str] = "src/robots/xarm/xarm7/xarm7.urdf"
XARM7_PACKAGE_DIR: Final[str] = "src/robots/xarm/xarm7"

# Default camera configuration
XARM7_DEFAULT_CAMERA_SPECS: Final[list[CameraSpec]] = CAMERAS


def rearrange_xarm7_deltas(delta: dict[str, float]) -> np.ndarray:
    """Rearrange deltas for xArm7 configuration."""
    return np.array(
        [
            delta.get("dx"),
            -delta.get("dz"),
            delta.get("dy"),
            delta.get("db"),
            delta.get("da"),
            delta.get("dg"),
        ]
    )


@RobotConfig.register_subclass("xarm7")
@dataclass(slots=True)
class XArm7Config(XArmConfig):
    """Configuration for UFactory xArm7 robot arm.

    The xArm7 is a 7-DOF collaborative robot arm with:
    - Reach: 700mm
    - Payload: 5kg
    - Repeatability: ±0.1mm
    - Weight: ~12kg
    """

    # Network configuration
    ip: str = XARM7_DEFAULT_IP

    # Robot specifications
    num_joints: int = 7  # xArm7 has 7 joints

    # Camera configuration
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            spec.name: spec.config for spec in XARM7_DEFAULT_CAMERA_SPECS
        }
    )

    # Start position (joint angles in degrees)
    start_pos: tuple[float, ...] = XARM7_SAFE_START_POS

    # Kinematics configuration
    urdf_path: str = XARM7_URDF_PATH
    package_dir: str = XARM7_PACKAGE_DIR
    tool_axis: str = XARM7_TOOL_AXIS
    tool_length: float = 0.165

    def init(self) -> None:
        super().init()

    def rearrange_deltas(self, delta: dict[str, float]) -> tuple[float, ...]:
        return rearrange_xarm7_deltas(delta)
