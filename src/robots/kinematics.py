import numpy as np
import pinocchio as pin
from typing import Literal
from pinocchio.pinocchio_pywrap_default import SE3, Frame, FrameType

from robots.joints import JointVelocities


class Kinematics:
    """
    Forward and inverse kinematics wrapper for the Mecademic Meca500 robot arm,
    using Pinocchio for model-based computation.

    Features:
      - Loads URDF and attaches an optional tool frame (TCP)
      - Computes forward kinematics (FK) to get the TCP pose
      - Computes iterative inverse kinematics (IK) to reach a desired pose
      - Computes joint velocities for small Cartesian deltas (Jacobian-based)

    Notes:
      * All kinematic quantities are expressed in the WORLD frame unless stated otherwise.
      * Angles are in radians, distances in meters.
      * Tool offset is automatically attached to the flange frame.
    """

    def __init__(
        self,
        urdf_path: str = "src/robots/meca/meca500r3.urdf",
        package_dir: str = "src/robots/meca",
        tool_length: float = 0.0,
        tool_axis: Literal["x", "y", "z"] = "x",
        tcp_name: str = "meca_tool_link",
        ee_link: str = "meca_axis_6_link",
    ):
        """
        Initialize the Meca500 kinematics model.

        Args:
            urdf_path: Path to the Meca500 URDF file.
            package_dir: Directory containing mesh and package resources.
            tool_length: Tool offset (in meters) from the flange along tool_axis.
            tool_axis: Direction of the tool offset ('x', 'y', or 'z').
            tcp_name: Name for the newly attached tool frame.

        Behavior:
            - Loads the robot model from URDF using Pinocchio.
            - Creates and attaches a tool frame at the flange offset.
            - Stores model and data structures for later FK/IK/Jacobian use.
        """
        # --- Load URDF model ---
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, package_dirs=[package_dir]
        )
        self.data = self.model.createData()

        # --- Add tool frame at flange ---
        flange_fid = self.model.getFrameId(ee_link)

        # Direction vector for the tool offset
        axis_map = {
            "x": np.array([1, 0, 0]),
            "y": np.array([0, 1, 0]),
            "z": np.array([0, 0, 1]),
        }
        axis_vec = axis_map.get(tool_axis, np.array([1, 0, 0]))
        tool_SE3 = SE3(np.eye(3), tool_length * axis_vec)

        tool_frame = Frame(
            tcp_name,
            self.model.frames[flange_fid].parentJoint,  # parent joint of flange
            flange_fid,  # attach after flange
            tool_SE3,
            FrameType.OP_FRAME,
        )

        # Add tool frame to the model and create updated data
        self.tcp_fid = self.model.addFrame(tool_frame)
        self.data = self.model.createData()

        print(
            f"✅ Added tool frame '{tcp_name}' at {tool_length:.3f} m along +{tool_axis.upper()} from flange."
        )

    def fk(self, q: np.ndarray) -> pin.pin.SE3:
        """
        Compute the forward kinematics of the robot.

        Args:
            q: Joint configuration (6x1 array in radians).

        Returns:
            pin.SE3: Pose of the tool frame (TCP) in the WORLD frame.
        """
        # Update joint placements and forward kinematics
        pin.pin.forwardKinematics(self.model, self.data, q)
        pin.pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.tcp_fid]

    # -------------------------------------------------------------------------
    # Iterative Inverse Kinematics (IK)
    # -------------------------------------------------------------------------
    def ik(
        self,
        q_init: np.ndarray,
        target_pose: pin.pin.SE3,
        iters: int = 100,
        tol: float = 0,
        damping: float = 0,
    ) -> np.ndarray:
        """
        Solve inverse kinematics iteratively to reach a target TCP pose.

        Args:
            q_init: Initial guess for joint configuration.
            target_pose: Desired end-effector pose as a Pinocchio SE3.
            iters: Maximum number of iterations.
            tol: Convergence tolerance on pose error.
            damping: Damping term for numerical stability (Tikhonov regularization).

        Returns:
            q: Final joint configuration achieving the desired pose (approximate).
        """
        q = q_init.copy()

        for _ in range(iters):
            pin.pin.forwardKinematics(self.model, self.data, q)
            pin.pin.updateFramePlacements(self.model, self.data)
            current_pose = self.data.oMf[self.tcp_fid]

            err_local = pin.log(current_pose.inverse() * target_pose)
            Ad = current_pose.action
            err_world = Ad @ err_local

            # Check convergence
            if np.linalg.norm(err_world) < tol:
                break

            J = pin.pin.computeFrameJacobian(
                self.model, self.data, q, self.tcp_fid, pin.pin.ReferenceFrame.WORLD
            )

            # Compute damped least-squares step
            JT = J.T
            dq = JT @ np.linalg.solve(J @ JT + damping * np.eye(6), err_world)
            q = pin.pin.integrate(self.model, q, dq)
        return q

    # -------------------------------------------------------------------------
    # Differential IK / Cartesian delta to joint velocities
    # -------------------------------------------------------------------------
    def compute_joint_velocities_from_delta(
        self,
        q: np.ndarray,
        delta: np.ndarray,
        dt: float = 0.016,
        damping: float = 1e-6,
    ) -> JointVelocities:
        """
        Compute joint velocities qdot (rad/s) that realize a small 6D motion
        where:
        - translation is expressed in LOCAL_WORLD_ALIGNED (world axes)
        - rotation is expressed in LOCAL (tool/body axes)
        """

        if delta.shape != (6,):
            raise ValueError("delta must have shape (6,)")

        # --- 1️⃣ Forward kinematics (ensure up to date)
        # pin.forwardKinematics(self.model, self.data, q)
        # pin.updateFramePlacements(self.model, self.data)

        # --- 2️⃣ Compute two Jacobians for different reference frames
        J_lwa = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.tcp_fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        J_local = pin.computeFrameJacobian(
            self.model, self.data, q, self.tcp_fid, pin.ReferenceFrame.LOCAL
        )

        # --- 3️⃣ Construct hybrid Jacobian:
        #     linear rows (0:3) from LOCAL_WORLD_ALIGNED
        #     angular rows (3:6) from LOCAL
        J_hybrid = np.vstack((J_lwa[:3, :], J_local[3:, :]))  # shape (6, n)

        # --- 4️⃣ Desired Cartesian twist (6×1): [vx, vy, vz, wx, wy, wz]
        v_cart = delta / dt

        # --- 5️⃣ Damped least-squares solve (qdot = Jᵀ (J Jᵀ + λI)⁻¹ v)
        JT = J_hybrid.T
        A = J_hybrid @ JT + damping * np.eye(6)
        qdot = JT @ np.linalg.solve(A, v_cart)

        return JointVelocities(qdot.astype(np.float32))
