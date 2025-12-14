from __future__ import annotations

import logging
from typing import Any, Final, Tuple

from lerobot.robots import Robot
import numpy as np
from utils.camera import make_non_blocking_cameras
from lerobot.processor.core import RobotAction, RobotObservation
from numpy.typing import NDArray

from robots.end_effector import EndEffectorPose
from robots.twoarm.config import TwoArmConfig
from robots.xarm.lite6.config import Lite6Config
from robots.xarm.robot import XArmRobot
from robots.xarm.xarm7.config import XArm7Config

FloatArray = NDArray[np.float32]

Frame = Tuple[int, int, int]  # (H, W, C)

Feature = dict[str, Any]


class TwoArm(Robot):
    config_class = TwoArmConfig
    name = "TwoArm"

    DEFAULT_FRAME: Final[Frame] = (360, 360, 3)

    # Camera read timeout (ms)
    CAM_READ_TIMEOUT_MS: Final[int] = 100

    log = logging.getLogger(__name__)

    def __init__(self, config: TwoArmConfig):
        self.camera_config = config.cameras
        self.cameras = make_non_blocking_cameras(config.cameras)
        self.left: Robot = config.left(config.left_config)
        self.right: Robot = config.right(config.right_config)

    def connect(self) -> None:
        for cam in self.cameras.values():
            try:
                cam.connect()
                self.log.info("Connected camera: %s", cam)
            except Exception:
                self.log.error("Failed to connect camera: %s", cam, exc_info=True)
        self.left.connect()
        self.right.connect()

    def disconnect(self) -> None:
        self.left.disconnect()
        self.right.disconnect()

    def calibrate(self) -> None:
        self.left.calibrate()
        self.right.calibrate()

    def reset(self) -> None:
        self.left.reset()
        self.right.reset()

    def configure(self) -> None:
        self.left.configure()
        self.right.configure()

    def get_ee_pose(self) -> EndEffectorPose:
        return {
            "right": self.right.get_ee_pose(),
            "left": self.left.get_ee_pose(),
        }

    def get_observation(self) -> RobotObservation:
        return {
            "right": self.right.get_observation(),
            "left": self.left.get_observation(),
        }

    def send_action(self, action: RobotAction) -> RobotAction:
        self.right.send_action(action["right"])
        self.left.send_action(action["left"])
        return action

    @property
    def observation_features(self) -> RobotObservation:
        """Structure of sensor outputs - input to policy"""
        ee_pose: Feature = {
            "right": {
                "x": float,
                "y": float,
                "z": float,
                "a": float,
                "b": float,
                "g": float,
                "gripper": float,
            },
            "left": {
                "x": float,
                "y": float,
                "z": float,
                "a": float,
                "b": float,
                "g": float,
                "gripper": float,
            },
        }
        camera_feature: Feature = {
            cam_name: self.DEFAULT_FRAME for cam_name in self.cameras.keys()
        }

        return {**ee_pose, **camera_feature}

    @property
    def action_features(self) -> RobotAction:
        """Structure of action - output of policy"""
        return {
            "right": {
                "dx": float,
                "dy": float,
                "dz": float,
                "da": float,
                "db": float,
                "dg": float,
                "gripper": float,
            },
            "left": {
                "dx": float,
                "dy": float,
                "dz": float,
                "da": float,
                "db": float,
                "dg": float,
                "gripper": float,
            },
        }

    @property
    def is_connected(self) -> bool:
        return self.left.is_connected and self.right.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.left.is_calibrated and self.right.is_calibrated

    def observe(self) -> RobotObservation:
        observation: RobotObservation = {}

        observation["left"] = self.left.get_ee_pose()
        observation["right"] = self.right.get_ee_pose()

        for cam_name, cam in self.cameras.items():
            frame: np.ndarray = cam.async_read(timeout_ms=self.CAM_READ_TIMEOUT_MS)
            assert frame is not None
            observation[cam_name] = frame

        return observation


if __name__ == "__main__":
    config = TwoArmConfig(
        left_config=Lite6Config(start_pos=(-59, 29, 56, 11.7, -24.2, -11), cameras={}),
        right_config=XArm7Config(
            start_pos=(1.7, -8.7, 17, 26, 103, 37, 27), cameras={}
        ),
        left=XArmRobot,
        right=XArmRobot,
        cameras={},
    )
    twoarm = TwoArm(config)
    twoarm.connect()
    twoarm.observe()
    twoarm.send_action(
        {
            "right": {
                "dx": 0,
                "dy": 0,
                "dz": 0,
                "da": 0,
                "db": 0,
                "dg": 0,
                "gripper": 0,
            },
            "left": {
                "dx": 0.1,
                "dy": 0,
                "dz": 0,
                "da": 0,
                "db": 0,
                "dg": 0,
                "gripper": 0,
            },
        }
    )
    twoarm.disconnect()
