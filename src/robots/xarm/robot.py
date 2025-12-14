from __future__ import annotations

from lerobot.robots import Robot
import numpy as np
from lerobot.cameras import make_cameras_from_configs
from lerobot.processor.core import RobotAction, RobotObservation
from robots.joints import JointPositions, JointVelocities
from robots.xarm.xarm import XArm

from utils.camera import make_non_blocking_cameras
from xarm.wrapper import XArmAPI

from typing_extensions import override

from robots.end_effector import EndEffectorPose
from robots.xarm.config import XArmConfig


class XArmRobot(XArm, Robot):
    config_class = XArmConfig
    name = "xarm"

    def __init__(self, config: XArmConfig):
        super().__init__(config)
        self.robot = XArmAPI(port=config.ip)
        self.camera_config = config.cameras
        self.cameras = make_non_blocking_cameras(config.cameras)
        self.num_joints = config.num_joints
        self.connected = False
        self.calibrated = False
        self.resetting = False
        self.gripper_state = 0  # Assume gripper starts open
        self.start_pos = config.start_pos
        self.ip = config.ip
        self.port = config.port

    @override
    def connect(self, calibrate: bool = True) -> None:
        self.resetting = True
        # self.robot.reset()
        self.robot.clean_error()
        self.robot.set_tcp_offset(offset=self.TOOL_TRF, is_rpy=False, wait=True)
        self.robot.motion_enable(enable=True)
        self.reset_position()


        self.log.info("Robot connected and ready.")

        for cam in self.cameras.values():
            try:
                cam.connect()
                self.log.info("Connected camera: %s", cam)
            except Exception:
                
                self.log.error("Failed to connect camera: %s trying again...", cam, exc_info=True)
                try:
                    cam.disconnect()
                    cam.connect()
                    self.log.info("Connected camera: %s", cam)
                except Exception:
                    self.log.error("Failed to connect camera: %s", cam, exc_info=True)

        self.connected = True
        self.resetting = False

    @override
    def disconnect(self) -> None:
        try:
            self.robot.disconnect()
        except Exception:
            self.log.error("Robot disconnect error.", exc_info=True)

        for cam in self.cameras.values():
            if getattr(cam, "is_connected", False):
                try:
                    cam.disconnect()
                except Exception:
                    self.log.error(
                        "Camera disconnect error for %s.", cam, exc_info=True
                    )
        self.connected = False
        self.log.info("Robot and cameras disconnected.")

    @override
    def get_ee_pose(self) -> EndEffectorPose:
        """Get the current end-effector pose as a 6D vector [x, y, z, a, b, g]."""

        pose = np.array(self.robot.get_position(is_radian=True)[1], dtype=np.float32)
        return EndEffectorPose(
            x=float(pose[0]),
            y=float(pose[1]),
            z=float(pose[2]),
            a=float(pose[3]),
            b=float(pose[4]),
            g=float(pose[5]),
        )

    @override
    def move(self, vels: JointVelocities) -> None:
        """Move joints with specified velocities (rad/s)."""

        self.robot.vc_set_joint_velocity(vels, is_radian=True, is_sync=False)

    @override
    def close_gripper(self) -> None:
        """Close the gripper."""
        self.robot.set_tgpio_digital(0, 1)

    @override
    def open_gripper(self) -> None:
        """Open the gripper."""
        self.robot.set_tgpio_digital(0, 0)

    @override
    def get_joint_positions(self) -> JointPositions:
        """Get the current joint positions as a numpy array."""
        return JointPositions(
            self.robot.get_servo_angle(is_radian=True)[1][: self.num_joints]
        )

    @override
    def get_observation(self) -> RobotObservation:
        return self.observe()

    @override
    def send_action(self, delta: RobotAction) -> RobotAction:
        return self.act(delta)

    @override
    def calibrate(self) -> None:
        pass

    @override
    def configure(self) -> None:
        pass

    def reset(self) -> None:
        if not self.connected:
            raise RuntimeError("Robot not connected. Call connect() before resetting.")

        self.reset_position()
        self.gripper_state = 0
        self.open_gripper()
        self.log.info("Robot reset and ready.")

    def pause(self) -> None:
        self.robot.set_mode(0)
        self.robot.set_state(3)

    def resume(self) -> None:
        self.robot.set_mode(4)
        self.robot.set_state(0)

    def robot_observation_processor(
        self, observation: RobotObservation
    ) -> RobotObservation:
        return observation


    def teleop_action_processor(self, action: tuple) -> dict:
        # Already in the right shape from OmniTeleoperator
        return action[0]

    def robot_action_processor(self, action: tuple) -> dict:
        # No processing needed for XArm7500
        return action[0]

    def reset_position(self) -> None:
        self.robot.set_mode(0)
        self.robot.set_state(0)

        self.robot.set_servo_angle(angle=self.start_pos, wait=True, is_radian=False)

        self.robot.set_mode(4)
        self.robot.set_state(0)

