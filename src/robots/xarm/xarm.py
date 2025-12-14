from __future__ import annotations

import logging
from typing import Any, Final, Tuple, cast

import numpy as np
from lerobot.processor.core import RobotAction, RobotObservation
from numpy.typing import NDArray

from robots.end_effector import EndEffectorDelta, EndEffectorPose
from robots.xarm.config import XArmConfig
from robots.joints import JointVelocities
from robots.kinematics import Kinematics

FloatArray = NDArray[np.float32]

Frame = Tuple[int, int, int]  # (H, W, C)

Feature = dict[str, Any]


class XArm:
    config_class = XArmConfig
    name = "xarm7"

    GRIPPER_FORCE: Final[int] = 10

    # Tool length of 148mm
    TOOL_TRF: Final[tuple[float, float, float, float, float, float]] = (
        0.0,
        0.0,
        148,
        0.0,
        0.0,
        0.0,
    )

    # Camera read timeout (ms)
    CAM_READ_TIMEOUT_MS: Final[int] = 200

    # Gripper thresholds
    GRIPPER_CLOSE_THR: Final[float] = 0.7
    GRIPPER_OPEN_THR: Final[float] = 0.3

    # Cartesian workspace (mm)
    WORKSPACE_LIMITS: Final[dict[str, tuple[float, float]]] = {
        "x": (-999, 999),
        "y": (-999, 999),
        "z": (-2, 240),  # only care about height for now
    }

    log = logging.getLogger(__name__)

    def __init__(self, config: XArmConfig):
        self.camera_config = config.cameras
        self.connected = False
        self.calibrated = False
        self.resetting = False
        self.gripper_state = 0  # Assume gripper starts open
        self.start_pos = config.start_pos
        self.kin = Kinematics(
            urdf_path=config.urdf_path,
            package_dir=config.package_dir,
            tool_axis=config.tool_axis,
            tool_length=config.tool_length,
            ee_link="link" + str(config.num_joints),
        )
        self.config = config

    @property
    def observation_features(self) -> RobotObservation:
        """Structure of sensor outputs - input to policy"""
        ee_pose: Feature = {
            "x": float,
            "y": float,
            "z": float,
            "a": float,
            "b": float,
            "g": float,
            "gripper": float,
        }
        camera_feature: Feature = {
            cam_name: (cam.config.height, cam.config.width, 3) for cam_name, cam in self.cameras.items()
        }

        return {**ee_pose, **camera_feature}

    @property
    def action_features(self) -> RobotAction:
        """Structure of action - output of policy"""
        return {
            "dx": float,
            "dy": float,
            "dz": float,
            "da": float,
            "db": float,
            "dg": float,
            "gripper": float,
        }

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def is_calibrated(self) -> bool:
        return self.calibrated

    def observe(self) -> RobotObservation:
        observation: RobotObservation = {}

        pose = self.get_ee_pose()
        observation.update(pose)
        observation["gripper"] = float(self.gripper_state)  # Add gripper state


        for cam_name, cam in self.cameras.items():
            frame = None
            frame: np.ndarray = cam.async_read()
            assert frame is not None
            observation[cam_name] = frame

        return observation

    def act(self, action: RobotAction) -> RobotAction:
        if self.resetting:
            self.log.debug("Ignoring action while resetting")
            return {}
        elif not self.connected:
            raise RuntimeError(
                "Robot not connected. Call connect() before sending actions."
            )

        current_pose: EndEffectorPose = self.get_ee_pose()
        delta: EndEffectorDelta = self._clamp_workspace(
            current_pose, self.coerce_action(action)
        )

        # translation in world frame
        vels = self.delta_to_velocity(delta)

        self.move(vels)

        self.grip(delta)

        return cast(RobotAction, delta)

    def coerce_action(self, action: RobotAction) -> EndEffectorDelta:
        # Throws on missing/non-numeric fields
        try:
            coerced: EndEffectorDelta = {
                "dx": float(action["dx"]),
                "dy": float(action["dy"]),
                "dz": float(action["dz"]),
                "da": float(action["da"]),
                "db": float(action["db"]),
                "dg": float(action["dg"]),
                "gripper": float(action["gripper"]),
            }
        except KeyError as e:
            raise KeyError(f"send_action: missing key '{e.args[0]}'")
        except (TypeError, ValueError) as e:
            raise ValueError(f"send_action: non-numeric field: {e}")

        return coerced

    def grip(self, delta: EndEffectorDelta):
        """Open / close the gripper if the value is sufficiently large or small"""

        gripper: float = delta.get("gripper")
        if gripper > 0.7 and self.gripper_state == 0:
            self.close_gripper()
            self.gripper_state = 1
        elif gripper < 0.3 and self.gripper_state == 1:
            self.open_gripper()
            self.gripper_state = 0

    def delta_to_velocity(self, delta: EndEffectorDelta) -> JointVelocities:
        rearranged = self.config.rearrange_deltas(delta)

        pos_deltas = rearranged[:3] / 1000  # Convert mm to m
        rot_deltas = rearranged[3:]

        damping = 1e-9

        return self.kin.compute_joint_velocities_from_delta(
            self.get_joint_positions(),
            np.concatenate([pos_deltas, rot_deltas]),
            dt=self.config.dt,
            damping=damping,
        )

    def move(self, vels: JointVelocities) -> None:
        """Move joints with specified velocities (rad/s)."""
        raise NotImplementedError

    def close_gripper(self) -> None:
        """Close the gripper."""
        raise NotImplementedError

    def open_gripper(self) -> None:
        """Open the gripper."""
        raise NotImplementedError

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions (rad)."""
        raise NotImplementedError

    def get_ee_pose(self):
        raise NotImplementedError

    def _clamp_workspace(
        self, pose: EndEffectorPose, delta: EndEffectorDelta
    ) -> EndEffectorDelta:
        """Zero out delta components that would exceed workspace limits."""

        return delta
