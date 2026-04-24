"""Position, velocity, and target-dummy helpers for the drone.

Thin wrappers around the CoppeliaSim remote API that return numpy
arrays suitable for direct concatenation into the RL observation
vector.  Also provides ``set_target`` to move the invisible target
dummy that the drone's PID controller follows.
"""

import numpy as np


def get_drone_pos_array(sim, drone_handle):
    """Gets the world-frame position of the drone as a numpy array.

    Args:
        sim: CoppeliaSim remote API object.
        drone_handle: Integer handle to the drone object.

    Returns:
        A float32 numpy array of shape ``(3,)`` containing
        ``[x, y, z]`` in world coordinates (metres).
    """
    pos = sim.getObjectPosition(drone_handle, sim.handle_world)
    return np.array(pos, dtype=np.float32)


def get_drone_velocity(sim, drone_handle):
    """Gets the linear velocity of the drone.

    Retrieves the current velocity vector from CoppeliaSim.
    Returns zeros if the velocity cannot be read.

    Args:
        sim: CoppeliaSim remote API object.
        drone_handle: Integer handle to the drone object.

    Returns:
        A float32 numpy array of shape ``(3,)`` containing
        ``[vx, vy, vz]`` in m/s.
    """
    result = sim.getObjectVelocity(drone_handle)
    linear_vel = result[0] if isinstance(result, (list, tuple)) else [0, 0, 0]
    return np.array(linear_vel, dtype=np.float32)


def set_target(sim, target_handle, x, y, z):
    """Sets the target dummy position in world coordinates.

    The drone's built-in PID controller tracks this dummy, so moving
    it is the mechanism by which the RL agent controls flight.

    Args:
        sim: CoppeliaSim remote API object.
        target_handle: Integer handle to the target dummy object.
        x: Target x-coordinate in metres.
        y: Target y-coordinate in metres.
        z: Target z-coordinate in metres.
    """
    sim.setObjectPosition(target_handle, sim.handle_world, [float(x), float(y), float(z)])
