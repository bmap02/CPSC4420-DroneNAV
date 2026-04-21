import numpy as np


def get_drone_pos_array(sim, drone_handle):
    """
    Gets the world-frame position of the drone as a numpy array.

    Identical to ``get_drone_pos`` but returns a numpy array
    for direct use in the RL observation vector.

    Args:
        sim: CoppeliaSim remote API object.
        drone_handle (int): Handle to the drone object.

    Returns:
        np.ndarray: Shape (3,) array of ``[x, y, z]`` in world coordinates.
    """
    pos = sim.getObjectPosition(drone_handle, sim.handle_world)
    return np.array(pos, dtype=np.float32)


def get_drone_velocity(sim, drone_handle):
    """
    Gets the linear velocity of the drone.

    Retrieves the current velocity vector from CoppeliaSim.
    Returns zeros if the velocity cannot be read.

    Args:
        sim: CoppeliaSim remote API object.
        drone_handle (int): Handle to the drone object.

    Returns:
        np.ndarray: Shape (3,) array of ``[vx, vy, vz]`` in m/s.
    """
    result = sim.getObjectVelocity(drone_handle)
    linear_vel = result[0] if isinstance(result, (list, tuple)) else [0, 0, 0]
    return np.array(linear_vel, dtype=np.float32)


def set_target(sim, target_handle, x, y, z):
    """
    Sets the target position in world coordinates.

    Args:
        sim: CoppeliaSim remote API object.
        target_handle (int): Handle to the target dummy object.
        x (float): Target x-coordinate in metres.
        y (float): Target y-coordinate in metres.
        z (float): Target z-coordinate in metres.
    """
    sim.setObjectPosition(target_handle, sim.handle_world, [float(x), float(y), float(z)])


