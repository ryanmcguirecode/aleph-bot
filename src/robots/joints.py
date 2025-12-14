from typing import Sequence
import numpy as np


class JointPositions(np.ndarray):
    """A NumPy array representing joint positions."""

    def __new__(cls, data: Sequence[float]):
        arr = np.asarray(data, dtype=float).view(cls)
        return arr


class JointVelocities(np.ndarray):
    """A NumPy array representing joint velocities."""

    def __new__(cls, data: Sequence[float]):
        arr = np.asarray(data, dtype=float).view(cls)
        return arr
