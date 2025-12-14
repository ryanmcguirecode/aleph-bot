import numpy as np


def eulerToQuat(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles (in radians) to quaternion."""
    cy = np.cos(euler[2] * 0.5)
    sy = np.sin(euler[2] * 0.5)
    cp = np.cos(euler[1] * 0.5)
    sp = np.sin(euler[1] * 0.5)
    cr = np.cos(euler[0] * 0.5)
    sr = np.sin(euler[0] * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qx, qy, qz, qw])


def quatToEuler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles (in radians)."""
    qx, qy, qz, qw = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])
