from typing import Callable
import threading
from pyOpenHaptics.hd import (
    start_scheduler,
    close_device,
    stop_scheduler,
    get_error,
    get_vendor,
    get_model,
    init_device,
)
from pyOpenHaptics.hd_callback import hdAsyncSheduler, hdSyncSheduler

"""Fork of pyOpenHaptics.HapticDevice with thread-safe initialization."""


class HapticDevice:
    _has_started_scheduler: bool = False
    _init_lock = threading.Lock()  # global lock for init_device
    callback: Callable[[], None]
    scheduler_type: str

    def __init__(
        self,
        callback: Callable[[], None],
        device_name: str = "Default Device",
        scheduler_type: str = "async",
    ):
        self.callback = callback
        self.scheduler_type = scheduler_type

        print(f"Initializing haptic device with name {device_name}")

        # Only one thread can call init_device at a time
        with HapticDevice._init_lock:
            self.id = init_device(device_name)

        print(f"Initialized device! {self.__vendor__()}/{self.__model__()}")

        if get_error():
            raise SystemError("Error initializing haptic device")

    def close(self):
        stop_scheduler()
        close_device(self.id)

    def scheduler(self, callback, scheduler_type):
        if scheduler_type == "async":
            hdAsyncSheduler(callback)
        else:
            hdSyncSheduler(callback)

    @staticmethod
    def __vendor__() -> str:
        return get_vendor()

    @staticmethod
    def __model__() -> str:
        return get_model()

    @staticmethod
    def initialize_scheduler():
        start_scheduler()

    def schedule(self):
        self.scheduler(self.callback, self.scheduler_type)
