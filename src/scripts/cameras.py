from contextlib import contextmanager
from typing import Dict, List, Optional

import cv2
from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

from cameraSpec import CAMERAS, CameraSpec, normalize


class CameraManager:
    """Manages multiple cameras and their lifecycle."""

    def __init__(self, camera_specs: List[CameraSpec]):
        self.camera_specs = camera_specs
        self.cameras: Dict[str, OpenCVCamera] = {}
        self._connected = False

    def connect_all(self) -> None:
        """Connect to all cameras."""
        for spec in self.camera_specs:
            camera = OpenCVCamera(spec.config)
            camera.connect()
            self.cameras[spec.name] = camera
        self._connected = True

    def disconnect_all(self) -> None:
        """Disconnect all cameras."""
        for camera in self.cameras.values():
            camera.disconnect()
        self.cameras.clear()
        self._connected = False

    def read_frames(self) -> Dict[str, Optional[any]]:
        """Read frames from all connected cameras and apply rotation if configured."""
        if not self._connected:
            raise RuntimeError("Cameras not connected. Call connect_all() first.")

        frames = {}
        for name, camera in self.cameras.items():
            frame = camera.async_read()
            if frame is None:
                frames[name] = None
                continue

            # Find camera spec
            spec = next(s for s in self.camera_specs if s.name == name)

            # Apply rotation from config (Cv2Rotation enum)
            rotation = getattr(spec.config, "rotation", Cv2Rotation.NO_ROTATION)
            if rotation == Cv2Rotation.ROTATE_90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == Cv2Rotation.ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == Cv2Rotation.ROTATE_270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Apply normalization (like color conversion)
            frame = normalize(frame, spec.config)

            frames[name] = frame

        return frames

    def get_camera_names(self) -> List[str]:
        """Get list of camera names."""
        return [spec.name for spec in self.camera_specs]

    @contextmanager
    def managed_connection(self):
        """Context manager for automatic camera connection/disconnection."""
        try:
            self.connect_all()
            yield self
        finally:
            self.disconnect_all()


def display_frames(frames: Dict[str, Optional[any]]) -> None:
    """Display frames in OpenCV windows."""
    for name, frame in frames.items():
        if frame is not None:
            # Convert RGB to BGR for OpenCV display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(f"Camera: {name}", display_frame)


def run_camera_viewer(camera_specs: List[CameraSpec]) -> None:
    """Run the camera viewer with the given camera specifications."""
    manager = CameraManager(camera_specs)

    try:
        with manager.managed_connection():
            print(f"Connected to cameras: {', '.join(manager.get_camera_names())}")
            print("Press 'q' to quit")

            while True:
                frames = manager.read_frames()
                display_frames(frames)

                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Configure which cameras to use
    cameras_to_use = CAMERAS

    # Run the camera viewer
    run_camera_viewer(cameras_to_use)
