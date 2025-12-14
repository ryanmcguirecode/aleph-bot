from __future__ import annotations

from lerobot.robots import Robot
import numpy as np
from lerobot.cameras import make_cameras_from_configs
from lerobot.processor.core import RobotAction, RobotObservation
from pyparsing import cast
from robots.joints import JointVelocities
from robots.meca.meca import Meca

from mecademicpy.mx_robot_def import MxEventSeverity, MxWorkZoneMode
from mecademicpy.robot import Robot as MecademicRobot
from typing_extensions import override

from cameraSpec import center_crop
from robots.end_effector import EndEffectorDelta, EndEffectorPose
from robots.meca.config import MecaConfig


class MecaRobot(Meca, Robot):
    config_class = MecaConfig
    name = "meca"

    def __init__(self, config: MecaConfig):
        super().__init__(config)
        self.robot = MecademicRobot()
        self.camera_config = config.cameras
        self.cameras = make_cameras_from_configs(config.cameras)
        self.connected = False
        self.calibrated = False
        self.resetting = False
        self.gripper_state = 0  # Assume gripper starts open
        self.start_pos = config.start_pos
        self.ip = config.ip

    @override
    def connect(self, calibrate: bool = True) -> None:
        self.resetting = True
        self.robot.Connect(self.ip)
        self.robot.ResetError()
        self.robot.SetTrf(*self.TOOL_TRF)

        if self.robot.GetStatusRobot(True).homing_state == 0:
            self.robot.SetToolSphere(0, 0, 148, 3)
            self.robot.SetWorkZoneLimits(-500, -500, 0, 500, 500, 500)
            # l=1 warning for out-of-zone, l=2 check tool only not joints
            self.robot.SetWorkZoneCfg(MxEventSeverity(1), MxWorkZoneMode(2))
            self.log.warning("Robot not homed; homing now.")
            self.robot.ActivateAndHome()
            self.robot.WaitHomed()
        else:
            self.robot.ResumeMotion()

        self.robot.SetJointVel(self.JOINT_VEL)
        self.robot.SetJointAcc(self.JOINT_ACC)
        self.robot.SetCartAcc(self.CART_ACC)
        self.robot.SetTorqueLimitsCfg(severity=1, skip_acceleration=True)
        self.robot.SetTorqueLimits(50.0, 50, 50, 50, 50, 50)

        self.robot.SetCartAngVel(self.CART_ANG_VEL)
        self.robot.SetGripperForce(self.GRIPPER_FORCE)

        self.robot.GripperOpen()
        if self.start_pos is not None:
            self.robot.MoveJoints(*self.start_pos)
        else:
            self.robot.MoveJoints(*self.REST_JOINTS)

        self.log.info("Robot connected and ready.")

        for cam in self.cameras.values():
            try:
                cam.connect()
                self.log.info("Connected camera: %s", cam)
            except Exception:
                self.log.error("Failed to connect camera: %s", cam, exc_info=True)

        self.connected = True
        self.resetting = False

    @override
    def disconnect(self) -> None:
        try:
            self.robot.Disconnect()
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

        pose = np.array(self.robot.GetRtTargetCartPos(), dtype=np.float32)
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

        # convert to degrees for mecademic API

        vels_degs = np.rad2deg(vels)

        # clamp velocities
        vels_degs = np.clip(vels_degs, -self.JOINT_VEL, self.JOINT_VEL)

        self.robot.MoveJointsVel(*vels_degs)

    @override
    def close_gripper(self) -> None:
        """Close the gripper."""
        self.robot.GripperClose()

    @override
    def open_gripper(self) -> None:
        """Open the gripper."""
        self.robot.GripperOpen()

    @override
    def get_joint_positions(self) -> np.ndarray:
        """Get the current joint positions as a numpy array."""
        return np.deg2rad(np.array(self.robot.GetRtTargetJointPos(), dtype=np.float32))

    @override
    def get_observation(self) -> RobotObservation:
        return self.observe()

    @override
    def send_action(self, action: RobotAction) -> RobotAction:
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

        # print(self.robot.GetRobotRtData().rt_joint_torq)

        vels = self.robot.MoveLinRelWRF(            
            -delta.get("dz"),
            -delta.get("dx"),
            delta.get("dy"),
            delta.get("dg"),
            delta.get("da"),
            -delta.get("db"),)

        self.grip(delta)
    
        return cast(RobotAction, delta)

    # @override
    # def send_action(self, delta: RobotAction) -> RobotAction:
    #     return self.act(delta)

    @override
    def calibrate(self) -> None:
        pass

    @override
    def configure(self) -> None:
        pass

    def reset(self) -> None:
        if not self.connected:
            raise RuntimeError("Robot not connected. Call connect() before resetting.")

        self.resetting = True

        self.robot.ResetError()
        self.robot.ActivateAndHome()
        self.robot.WaitHomed()
        self.robot.ResumeMotion()

        self.reset_position()

        self.resetting = False
        self.log.info("Robot reset and ready.")

    def reset_position(self) -> None:
        self.resetting = True
        self.robot.MoveJoints(*self.REST_JOINTS)
        self.robot.WaitIdle()
        self.resetting = False
        self.robot.GripperOpen()
        self.gripper_state = 0

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
