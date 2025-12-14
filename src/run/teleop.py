import logging
from time import perf_counter

import cv2
from lerobot.robots import Robot

from teleoperators.controller import Controller
from gui import Gui
from .experiment import Teleop
from timing import wait_for_next_tick

log = logging.getLogger("teleop")


def teleop(experiment: Teleop):

    gui: Gui = Gui()

    robot: Robot = experiment.robot
    controller: Controller = experiment.controller

    target_hz: float = experiment.target_hz
    dt: float = 1.0 / target_hz
    tick: float = perf_counter()

    # Speed measurement
    loop_count: int = 0
    speed_start_time: float = perf_counter()
    speed_report_interval: float = 1.0  # Report speed every 1 second

    try:
        robot.connect()
        controller.connect()

        log.info("Teleop loop started. Press 'q' to quit.")

        while True:
            action = controller.get_action()
            robot.send_action(action)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            tick = wait_for_next_tick(tick, dt)
            
            # Track speed
            loop_count += 1
            elapsed = perf_counter() - speed_start_time
            if elapsed >= speed_report_interval:
                empirical_hz = loop_count / elapsed
                log.info(f"Empirical speed: {empirical_hz:.2f} Hz (target: {target_hz:.2f} Hz)")
                loop_count = 0
                speed_start_time = perf_counter()

    except KeyboardInterrupt:
        log.info("Stopping teleoperation (Ctrl+C).")
    finally:
        try:
            teleop.disconnect()
        except Exception:
            log.error("Failed to disconnect controller")

        try:
            robot.disconnect()
        except Exception:
            log.error("Failed to disconnect robot")

        gui.safe_close()

        log.info("Shutdown complete.")
