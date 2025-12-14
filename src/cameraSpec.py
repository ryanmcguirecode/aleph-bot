from dataclasses import dataclass
from typing import Tuple

from cv2 import flip
from lerobot.cameras import CameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig
from numpy import ndarray


@dataclass(frozen=True, slots=True)
class CameraSpec:
    name: str
    config: CameraConfig


FrameSize = Tuple[int, int]


WRIST_1: CameraSpec = CameraSpec(
    name="wrist 1",
    config=OpenCVCameraConfig(
        index_or_path=1,
        width=1280,
        height=720,
        fps=90,
        fourcc="MJPG",
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
    ),
)

WRIST_2: CameraSpec = CameraSpec(
    name="wrist 2",
    config=OpenCVCameraConfig(
        index_or_path=8,
        width=1280,
        height=720,
        fps=90,
        fourcc="MJPG",
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
    ),
)

SURGEON_VIEW = CameraSpec(
    name="surgeon view",
    config=OpenCVCameraConfig(
        index_or_path=6,
        width=640,
        height=480,
        fps=30,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
    ),
)

FRONT_VIEW = CameraSpec(
    name="front view",
    config=OpenCVCameraConfig(
        index_or_path=0,
        width=1280,
        height=720,
        fps=120,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
    ),
)

MICROSCOPE_LEFT = CameraSpec(
    name="microscope left",
    config=OpenCVCameraConfig(
        index_or_path=12,
        width=1280,
        height=720,
        fps=30,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
    ),
)

MICROSCOPE_RIGHT = CameraSpec(
    name="microscope right",
    config=OpenCVCameraConfig(
        index_or_path=14,
        width=1280,
        height=720,
        fps=30,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
    ),
)
CAMERAS = [WRIST_1, WRIST_2, SURGEON_VIEW, FRONT_VIEW, MICROSCOPE_LEFT, MICROSCOPE_RIGHT]
CAMERAS_DICT = {
    "WRIST_1": WRIST_1.config,
    "WRIST_2": WRIST_2.config,
    # "SURGEON_VIEW": SURGEON_VIEW.config,
    "FRONT_VIEW": FRONT_VIEW.config,
    # "MICROSCOPE_LEFT": MICROSCOPE_LEFT.config,
    # "MICROSCOPE_RIGHT": MICROSCOPE_RIGHT.config,
}

def center_crop(img: ndarray, output_size: FrameSize = (256, 256)) -> ndarray:
    """
    Center crop + resize an image.
    Args:
        img: numpy array (H, W, C)
        output_size: tuple (h, w) for final size
    Returns:
        Cropped and resized numpy array
    """
    h, w, _ = img.shape
    out_h, out_w = output_size

    # Ensure the crop doesn’t exceed the original image
    crop_h = min(out_h, h)
    crop_w = min(out_w, w)

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    cropped = img[top : top + crop_h, left : left + crop_w]
    return cropped


def normalize(img: ndarray, config: CameraConfig) -> ndarray:
    """Normalize the image for the given CameraConfig (e.g. undo rotations)"""

    if (
        isinstance(config, OpenCVCameraConfig)
        and config.rotation == Cv2Rotation.ROTATE_180
    ):
        return flip(img, -1)

    return img
