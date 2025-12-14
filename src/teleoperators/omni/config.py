from dataclasses import dataclass
from pathlib import Path
from typing import Final

from lerobot.teleoperators.config import TeleoperatorConfig

DEVICE_NAME: Final[str] = "USB"  # USB or LAN

TRANSLATION_SCALES: Final[tuple[float, ...]] = (0.1, 0.4, 1.0)

SCALE_ROTATION: Final[float] = 1

MAX_STEP: Final[float] = 2.0

MAX_ANGLE: Final[float] = 2.0

# Low-pass filter settings (None to disable)
CUTOFF_FREQUENCY_HZ: Final[float | None] = 20  # Hz threshold above which frequencies are removed
SAMPLING_RATE_HZ: Final[float] = 250.0  # Sampling rate for the filter


@dataclass
class OmniConfig(TeleoperatorConfig):
    id = "0"
    path = Path("~/.lerobot/calibration/omni").expanduser()

    device_name: str = DEVICE_NAME
    translation_scales: tuple[float, ...] = TRANSLATION_SCALES
    scale_rotation: float = SCALE_ROTATION
    max_step: float = MAX_STEP
    max_angle: float = MAX_ANGLE
    buttons: Final[tuple[str, ...]] = ("GRIPPER", "SCALE")
    cutoff_frequency_hz: float | None = CUTOFF_FREQUENCY_HZ
    sampling_rate_hz: float = SAMPLING_RATE_HZ
