from __future__ import annotations
import time

from lerobot.robots import Robot
import numpy as np
from lerobot.cameras import make_cameras_from_configs
from lerobot.processor.core import RobotAction, RobotObservation
from pyparsing import cast
from robots.fairinofr3.fairinofr3 import Fairino

from fairino import RPC
from typing_extensions import override

from cameraSpec import center_crop
from robots.end_effector import EndEffectorDelta, EndEffectorPose
from robots.fairinofr3.config import FairinoConfig


class FairinoRobot(Fairino, Robot):
    config_class = FairinoConfig
    name = "fairino"

    def __init__(self, config: FairinoConfig):
        super().__init__(config)
        self.robot = RPC(config.ip)
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
        self.robot.RobotEnable(1)
        self.robot.Mode(0)

        err = self.robot.SetToolList(
            id=1,
            t_coord=self.TOOL_TRF,
            type=0,  # tool
            install=0,  # mounted on flange
            loadNum=0,  # default payload
        )

        if err == 0:
            print("Tool 1 set successfully!")
        else:
            print("SetToolList error code:", err)

        self.robot.ServoMoveStart()

        # if self.robot.GetStatusRobot(True).homing_state == 0:
        #     self.robot.SetToolSphere(0, 0, 148, 3)
        #     self.robot.SetWorkZoneLimits(-500, -500, 0, 500, 500, 500)
        #     # l=1 warning for out-of-zone, l=2 check tool only not joints
        #     self.robot.SetWorkZoneCfg(MxEventSeverity(1), MxWorkZoneMode(2))
        #     self.log.warning("Robot not homed; homing now.")
        #     self.robot.ActivateAndHome()
        #     self.robot.WaitHomed()
        # else:
        #     self.robot.ResumeMotion()

        # self.robot.SetJointVel(self.JOINT_VEL)
        # self.robot.SetJointAcc(self.JOINT_ACC)
        # self.robot.SetCartAcc(self.CART_ACC)
        # self.robot.SetTorqueLimitsCfg(severity=1, skip_acceleration=True)
        # self.robot.SetTorqueLimits(50.0, 50, 50, 50, 50, 50)

        # self.robot.SetCartAngVel(self.CART_ANG_VEL)
        # self.robot.SetGripperForce(self.GRIPPER_FORCE)

        # self.robot.GripperOpen()
        if self.start_pos is not None:
            self.robot.MoveJ(
                joint_pos=self.start_pos,
                tool=1,  # active tool
                user=0,  # base frame
                vel=10.0,  # 50% velocity
                acc=0.0,
                ovl=100.0,
                blendT=-1.0,  # blocking
                offset_flag=0,
                offset_pos=[0, 0, 0, 0, 0, 0],
            )

        else:
            self.robot.MoveJ(
                joint_pos=self.REST_JOINTS,
                tool=1,  # active tool
                user=0,  # base frame
                vel=10.0,  # 50% velocity
                acc=0.0,
                ovl=100.0,
                blendT=-1.0,  # blocking
                offset_flag=0,
                offset_pos=[0, 0, 0, 0, 0, 0],
            )

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

        pose = np.array(self.robot.GetActualTCPPose()[1], dtype=np.float32)
        return EndEffectorPose(
            x=float(pose[0]),
            y=float(pose[1]),
            z=float(pose[2]),
            a=float(pose[3]),
            b=float(pose[4]),
            g=float(pose[5]),
        )

    # @override
    # def move(self, vels: JointVelocities) -> None:
    #     """Move joints with specified velocities (rad/s)."""

    #     # convert to degrees for mecademic API

    #     vels_degs = np.rad2deg(vels)

    #     #clamp velocities
    #     vels_degs = np.clip(vels_degs, -self.JOINT_VEL, self.JOINT_VEL)

    #     self.robot.MoveJointsVel(*vels_degs)

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

        # dpose = [
        #     -delta.get("dz", 0.0),   # X (mm)
        #     -delta.get("dx", 0.0),   # Y (mm)
        #     delta.get("dy", 0.0),    # Z (mm)
        #     delta.get("da", 0.0),    # Rx (deg)
        #     delta.get("db", 0.0),    # Ry (deg)
        #     delta.get("dg", 0.0),   # Rz (deg)
        # ]

        dpose_move = [
            -delta.get("dx", 0.0),  # Y (mm)
            delta.get("dz", 0.0),  # X (mm)
            delta.get("dy", 0.0),  # Z (mm)
            0.0,  # Rx (deg)
            0.0,  # Ry (deg)
            0.0,  # Rz (deg)
        ]

        dpose_rot = [
            0.0,  # X (mm)
            0.0,  # Y (mm)
            0.0,  # Z (mm)
            -delta.get("db", 0.0),  # Ry (deg)
            -delta.get("da", 0.0),  # Rx (deg)
            delta.get("dg", 0.0),  # Rz (deg)
        ]

        # Typical gains and timing
        pos_gain = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        cmdT = 0.004  # 8 ms control cycle

        # move in world frame
        err = self.robot.ServoCart(1, dpose_move, pos_gain, 0.0, 0.0, cmdT, 0.0, 0.0)
        if err != 0:
            print(f"ServoCart error: {err}")

        time.sleep(0.002)

        # rotate in tool frame
        err = self.robot.ServoCart(2, dpose_rot, pos_gain, 0.0, 0.0, cmdT, 0.0, 0.0)
        if err != 0:
            print(f"ServoCartTool error: {err}")

        self.grip(delta)

        return cast(RobotAction, delta)

    @override
    def close_gripper(self) -> None:
        """Close the gripper."""
        # self.robot.GripperClose()
        pass

    @override
    def open_gripper(self) -> None:
        """Open the gripper."""
        # self.robot.GripperOpen()
        pass

    @override
    def get_joint_positions(self) -> np.ndarray:
        """Get the current joint positions as a numpy array."""
        return np.deg2rad(
            np.array(self.robot.GetActualJointPosDegree(0), dtype=np.float32)
        )

    @override
    def get_observation(self) -> RobotObservation:
        return self.observe()

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
