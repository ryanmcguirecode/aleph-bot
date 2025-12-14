from typing import TypedDict


class EndEffectorPose(TypedDict):
    x: float
    y: float
    z: float
    a: float
    b: float
    g: float


class EndEffector(TypedDict):
    position: EndEffectorPose
    gripper: float


# Action that the policy outputs
class EndEffectorDelta(TypedDict):
    dx: float
    dy: float
    dz: float
    da: float
    db: float
    dg: float
    gripper: float


# 6D end-effector velocity in Cartesian space (vx, vy, vz, va, vb, vg) in rad/s and m/s.
class EndEffectorVelocity(TypedDict):
    vx: float
    vy: float
    vz: float
    va: float
    vb: float
    vg: float


def add_delta(position: EndEffectorPose, delta: EndEffectorDelta) -> EndEffectorPose:
    return EndEffectorPose(
        x=(position.get("x") + delta.get("dx")),
        y=(position.get("y") + delta.get("dy")),
        z=(position.get("z") + delta.get("dz")),
        a=(position.get("a") + delta.get("da")),
        b=(position.get("b") + delta.get("db")),
        g=(position.get("g") + delta.get("dg")),
    )
