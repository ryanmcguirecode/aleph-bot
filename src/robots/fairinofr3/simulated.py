from __future__ import annotations
from typing import Any


from lerobot.robots import Robot
import numpy as np
from lerobot.processor.core import RobotAction, RobotObservation
from typing_extensions import override

from cameraSpec import center_crop
from robots.end_effector import EndEffectorPose
from robots.fairino.config import FairinoConfig
from robots.fairino.fairino import Fairino
from robots.joints import JointVelocities


import genesis as gs

from utils.spatial import quatToEuler


KP = np.array([253.0, 214.0, 557.0, 200.0, 1500.0, 900.0])
KV = np.array([11.0, 14.0, 20.0, 12.0, 50.0, 22.0])


class FairinoSim(Fairino, Robot):
    config_class = FairinoConfig
    name = "fairinosim"

    def __init__(self, config: FairinoConfig) -> None:
        super().__init__(config)
        gs.init(
            logging_level="warning",
        )
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.01,
                gravity=(0, 0, 0),
            ),
            show_viewer=True,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.50, 0.0, 1.50),
                camera_lookat=(0.2, 0, 0.5),
                camera_fov=40,
                max_FPS=60,
                res=(1280, 720),
            ),
            vis_options=gs.options.VisOptions(
                ambient_light=(0.5, 0.5, 0.5),
                show_world_frame=True,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,  # Enable self-collision for rigid bodies
            ),
        )

        self.scene.add_entity(gs.morphs.Plane())

        self.robot: Any = self.scene.add_entity(
            gs.morphs.URDF(
                file="src/robots/fairino/fairino3_v6.urdf",
                pos=(0, 0, 0),  # Position the Fairino robot above the plane
                euler=(0, 0, 90),
                fixed=True,
                collision=False,
            ),
        )

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def is_calibrated(self) -> bool:
        return self.calibrated

    @override
    def connect(self, calibrate: bool = True) -> None:
        self.scene.build()

        # Filter out joints with a valid DOF index (non-fixed)
        movable_joints = [
            joint for joint in self.robot.joints if joint.dof_idx_local is not None
        ]

        # Extract names and DOF indices
        # jnt_names = [joint.name for joint in movable_joints]
        dofs_idx = [joint.dof_idx_local for joint in movable_joints][:7]

        # Set control gains per DOF
        self.robot.set_dofs_kp(KP, dofs_idx)
        self.robot.set_dofs_kv(KV, dofs_idx)
        self.robot.set_dofs_position(
            (
                np.radians(self.start_pos)
                if self.start_pos
                else np.radians(self.REST_JOINTS)
            ),
            dofs_idx,
        )
        for _ in range(100):
            self.scene.step()
        self.connected = True

    @override
    def send_action(self, action: RobotAction) -> RobotAction:
        return self.act(action)

    @override
    def get_ee_pose(self) -> EndEffectorPose:
        """Get the current end-effector pose as a 6D vector [x, y, z, a, b, g]."""
        pos = self.robot.get_link("wrist3_link").get_pos().cpu().numpy()
        quat = self.robot.get_link("wrist3_link").get_quat().cpu().numpy()
        rot = quatToEuler(quat)
        return EndEffectorPose(
            x=pos[0],
            y=pos[1],
            z=pos[2],
            a=rot[0],
            b=rot[1],
            g=rot[2],
        )

    @override
    def calibrate(self) -> None:
        raise NotImplementedError

    @override
    def configure(self) -> None:
        raise NotImplementedError

    @override
    def get_observation(self) -> RobotAction:
        return {}

    @override
    def disconnect(self) -> None:
        raise NotImplementedError

    def get_joint_positions(self) -> np.ndarray:
        """Get the current joint positions as a numpy array."""
        return self.robot.get_qpos().cpu().numpy()

    @override
    def move(self, vels: JointVelocities) -> None:
        """Send joint velocities (rad/s) to the robot."""
        # DICT TO LIST
        self.scene.step()
        # SEND VELOCITIES
        self.robot.control_dofs_velocity(vels)

    def robot_observation_processor(
        self, observation: RobotObservation
    ) -> RobotObservation:
        processed = observation.copy()

        for camera_name, camera_config in self.camera_config.items():
            assert camera_name in observation
            processed[camera_name] = center_crop(
                observation[camera_name], self.DEFAULT_FRAME[:2]
            )

        return processed

    def teleop_action_processor(self, action: tuple) -> dict:
        # Already in the right shape from OmniTeleoperator
        return action[0]

    def robot_action_processor(self, action: tuple) -> dict:
        # No processing needed for Meca500
        return action[0]
