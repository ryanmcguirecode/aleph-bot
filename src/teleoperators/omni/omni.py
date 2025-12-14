import logging
import time
import weakref
from ctypes import c_double
from dataclasses import dataclass
from typing import Any, Callable, TypeAlias

import numpy as np
from numpy import array, ndarray
from numpy.typing import NDArray
from pyOpenHaptics.hd import get_buttons, get_transform, start_scheduler, stop_scheduler
from pyOpenHaptics.hd_callback import hd_callback
from teleoperators.omni.hd_device import HapticDevice
from scipy.spatial.transform import Rotation
from scipy.signal import butter, lfilter
from typing_extensions import override

from teleoperators.controller import Controller
from teleoperators.omni.buttons import OmniButtons
from teleoperators.omni.config import OmniConfig

log = logging.getLogger(__name__)

Row4 = c_double * 4
TransformC: TypeAlias = Row4 * 4  # type: ignore

Vec3 = NDArray[np.float64]
Mat44 = NDArray[np.float64]


class TremorFilter:
    """Simple exponential low-pass filter for tremor suppression."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.prev = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = x.copy()
            return x
        self.prev = self.alpha * x + (1 - self.alpha) * self.prev
        return self.prev


class LowPassFilter:
    """Butterworth low-pass filter that removes frequencies above a threshold."""

    def __init__(self, cutoff_hz: float, sampling_rate_hz: float, order: int = 2):
        """
        Args:
            cutoff_hz: Cutoff frequency in Hz (frequencies above this are removed)
            sampling_rate_hz: Sampling rate in Hz
            order: Filter order (higher = steeper rolloff)
        """
        self.cutoff_hz = cutoff_hz
        self.sampling_rate_hz = sampling_rate_hz
        self.order = order
        
        # Design Butterworth low-pass filter
        nyquist = sampling_rate_hz / 2.0
        normal_cutoff = cutoff_hz / nyquist
        self.b, self.a = butter(order, normal_cutoff, btype="low", analog=False)
        
        # Initialize filter state for each channel (6 DOF: dx, dy, dz, da, db, dg)
        self.zi = None
        self.initialized = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Filter the input signal (single sample, 1D array).
        
        Args:
            x: Input signal (1D array of 6 values: dx, dy, dz, da, db, dg)
            
        Returns:
            Filtered signal with same shape as input
        """
        if not self.initialized:
            # Initialize filter state for each channel
            # zi shape: (max(len(a), len(b)) - 1, num_channels)
            zi_shape = (max(len(self.a), len(self.b)) - 1,)
            self.zi = np.zeros((zi_shape[0], len(x)))
            self.initialized = True
        
        # Apply filter to each channel independently
        # x is 1D array, we need to filter each element
        filtered = np.zeros_like(x)
        for i in range(len(x)):
            # Filter single sample: need to reshape for lfilter
            sample = np.array([x[i]])
            filtered_sample, self.zi[:, i] = lfilter(
                self.b, self.a, sample, zi=self.zi[:, i]
            )
            filtered[i] = filtered_sample[0]
        
        return filtered
    
    def reset(self):
        """Reset filter state."""
        self.zi = None
        self.initialized = False


@dataclass
class DeviceState:
    transform: Mat44 = np.eye(4)
    buttons: int = 0


class OmniTeleoperator(Controller):
    config_class = OmniConfig
    name = "omni"
    _scheduled = False
    _initialized = False

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def feedback_features(self) -> dict:
        """No haptic feedback features exposed yet."""
        return {}

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "dx": float,
            "dy": float,
            "dz": float,
            "da": float,
            "db": float,
            "dg": float,
            "gripper": bool,
        }

    def __init__(self, config: OmniConfig):
        super().__init__(config)
        self.cb = self._make_device_cb()
        self.buttons: OmniButtons = OmniButtons(
            scales=config.translation_scales, buttons=config.buttons
        )
        self.device: HapticDevice = HapticDevice(
            callback=self.cb, scheduler_type="async", device_name=config.device_name
        )
        self.device.initialize_scheduler()
        self.device.schedule()
        self.device_state: DeviceState = DeviceState()

        self.config = config

        self.position = None
        self.rotation = None

        self.connected = True
        self.warmup = 10  # ignore first N readings to stabilize

        self.last_poll_time = time.perf_counter()  #
        self.translation_filter = TremorFilter(
            alpha=1
        )  # Strong smoothing for translation
        self.rotation_filter = TremorFilter(
            alpha=1
        )  # Even stronger smoothing for rotation
        
        # Initialize low-pass filter if cutoff frequency is specified
        self.lowpass_filter = None
        if config.cutoff_frequency_hz is not None:
            self.lowpass_filter = LowPassFilter(
                cutoff_hz=config.cutoff_frequency_hz,
                sampling_rate_hz=config.sampling_rate_hz,
                order=2,
            )
            log.info(
                f"Low-pass filter enabled: cutoff={config.cutoff_frequency_hz} Hz, "
                f"sampling_rate={config.sampling_rate_hz} Hz"
            )
        
        stop_scheduler()

    def _check_stale(self, timeout: float = 0.1) -> bool:
        """Return True if the haptic device hasn't sent updates recently."""
        return (time.perf_counter() - self.last_poll_time) > timeout

    def reset(self) -> None:
        self.position = None
        self.rotation = None
        self.warmup = 10
        # Reset tremor filters when position is reset
        self.translation_filter.prev = None
        self.rotation_filter.prev = None
        # Reset low-pass filter if enabled
        if self.lowpass_filter is not None:
            self.lowpass_filter.reset()

    @override
    def connect(self, calibrate: bool = True) -> None:
        log.info("Connected to omni controller")
        if not OmniTeleoperator._scheduled:
            start_scheduler()
        else:
            print("Scheduler already started")
        OmniTeleoperator._scheduled = True
        self.connected = True

    @override
    def disconnect(self) -> None:
        log.info("Disconnecting from omni controller")
        if self.connected:
            self.device.close()
            self.connected = False

    @override
    def get_action(self) -> dict[str, Any]:
        if self._check_stale():
            log.warning("⚠️  Haptic device not polled in >0.1s — resetting.")
            self.reset()
        self.last_poll_time = time.perf_counter()  # update heartbeat
        state = self._get_device_state()

        _position: ndarray = state[0]
        T: ndarray = state[1]
        buttons_raw: int = state[2]

        update = self.buttons.update(buttons_raw)

        deltas, self.position, self.rotation = self._extract_deltas(
            T,
            self.config.scale_rotation,
            update.motion_scale,
        )
        
        # Apply low-pass filter if enabled (removes frequencies above threshold)
        if self.lowpass_filter is not None:
            deltas_array = np.array([deltas[0], deltas[1], deltas[2], deltas[3], deltas[4], deltas[5]])
            filtered_deltas = self.lowpass_filter(deltas_array)
            dx, dy, dz, da, db, dg = (
                filtered_deltas[0],
                filtered_deltas[1],
                filtered_deltas[2],
                filtered_deltas[3],
                filtered_deltas[4],
                filtered_deltas[5],
            )
        else:
            dx, dy, dz, da, db, dg = deltas

        return {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "da": da,
            "db": db,
            "dg": dg,
            "gripper": update.gripper_state.as_bool,
        }

    @override
    def configure(self) -> None:
        """No configuration needed for Omni."""
        pass

    @override
    def calibrate(self) -> None:
        """No calibration needed for Omni."""
        pass

    @override
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # Optionally implement force feedback if pyOpenHaptics exposes it
        pass

    @override
    def motion_scale(self) -> float:
        return self.buttons.motion_scale

    def set_motion_scale(self, scale: float) -> None:
        self.buttons.set_motion_scale(scale)

    def _extract_deltas(self, matrix, scale_rotation: float, scale_translation: float):
        """Compute relative translation and orientation deltas from haptic transform."""
        T = array([[matrix[i][j] for j in range(4)] for i in range(4)])
        position = T[3, :3]  # last row = position
        rotation_matrix = T[:3, :3]  # 3x3 rotation

        if self.position is None or self.rotation is None or self.warmup > 0:

            self.position = position.copy()
            self.rotation = rotation_matrix.copy()

            if self.warmup > 0:
                self.warmup -= 1
            return (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                position.copy(),
                rotation_matrix.copy(),
            )

        dh = (position - self.position) * scale_translation
        translation_delta = np.array([dh[0], dh[1], dh[2]])

        # Apply tremor filter to translation deltas
        translation_filtered = self.translation_filter(translation_delta)
        dx, dy, dz = (
            translation_filtered[0],
            translation_filtered[1],
            translation_filtered[2],
        )

        dR = rotation_matrix @ self.rotation.T
        rotvec = Rotation.from_matrix(dR).as_rotvec() * scale_rotation

        # Apply tremor filter to rotation deltas
        rotation_filtered = self.rotation_filter(rotvec)
        dα, dβ, dγ = rotation_filtered[0], rotation_filtered[1], rotation_filtered[2]

        return (dx, dy, dz, dα, dβ, dγ), position.copy(), rotation_matrix.copy()

    def _make_device_cb(self) -> Callable[[], None]:
        owner_ref = weakref.ref(self)

        @hd_callback
        def _cb() -> None:
            owner = owner_ref()
            if owner is None or not owner.connected:
                return
            owner.device_state.transform = owner._to_np(get_transform())
            owner.device_state.buttons = int(get_buttons())

        return _cb

    def _get_device_state(self) -> tuple[Vec3, Mat44, int]:
        pos: Vec3 = self.device_state.transform[:3, 3].copy()
        return pos, self.device_state.transform.copy(), int(self.device_state.buttons)

    @staticmethod
    def _to_np(ct_T: TransformC) -> Mat44:
        # Convert ctypes (c_double[4][4]) -> np.ndarray (4,4), float64
        return np.ctypeslib.as_array(ct_T, shape=(4, 4)).copy()


def main() -> None:
    # config = OmniConfig(device_name="LAN")
    # omni_right = OmniTeleoperator(config)
    omni_left = OmniTeleoperator(
        OmniConfig(
            translation_scales=(0.1, 0.4, 1.0),
            scale_rotation=1.0,
            device_name="LAN",
        )
    )
    # omni_right.connect()
    omni_left.connect()
    try:
        while True:
            print(omni_left.get_action())
            # omni_right.get_action()
            time.sleep(0.009)
    except KeyboardInterrupt:
        pass
    finally:
        omni_left.disconnect()
        # omni_right.disconnect()


if __name__ == "__main__":
    main()
