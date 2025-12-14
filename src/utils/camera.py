import threading
import time
from lerobot.cameras import Camera, CameraConfig
from lerobot.cameras.opencv import OpenCVCamera
from numpy.typing import NDArray
from typing import Any

class NonBlockingCamera(OpenCVCamera):
    """
    Wraps any LeRobot camera to make async reads non-blocking.
    Returns the latest cached frame immediately, reusing the last one if no new frame is available.
    """

    def __init__(self, config: CameraConfig):
        super().__init__(config)
    def read_latest(self) -> NDArray[Any]:
        """Return the most recent frame instantly (reuses last if none new)."""

        if not self.is_connected:
            raise RuntimeError(f"Camera {self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()
            time.sleep(0.1)

        with self.frame_lock:
            frame = self.latest_frame
        if frame is not None:
            with self.frame_lock:
                self.latest_frame = frame
        if self.latest_frame is None:
            raise RuntimeError("No frame available yet.")
        return self.latest_frame

    def async_read(self):
        return self.read_latest()


def make_non_blocking_cameras(cameras: dict[str, Camera]) -> dict[str, NonBlockingCamera]:
    return {name: NonBlockingCamera(config) for name, config in cameras.items()}