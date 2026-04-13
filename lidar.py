import numpy as np


# ── Default sensor paths for all 4 LiDAR sensors ──
LIDAR_SENSOR_PATHS = {
    'F': '/Quadcopter/base/lidarFront/body/sensor',   # Front sensor
    'B': '/Quadcopter/base/lidarBack/body/sensor',     # Back sensor
    'L': '/Quadcopter/base/lidarLeft/body/sensor',     # Left sensor
    'R': '/Quadcopter/base/lidarRight/body/sensor',    # Right sensor
}

DEFAULT_MAX_RANGE = 5.0                                # Return value when no valid reading (m)


def set_lidar_resolution(sim, lidar_handles, resolution):
    """
    Attempts to set the resolution of each LiDAR vision sensor.

    Note: This depends on the CoppeliaSim API exposing the
    vision sensor integer params in the remote API.

    Args:
        sim: CoppeliaSim remote API object.
        lidar_handles (dict[str, int]): Sensor handle map.
        resolution (int | None): Desired square resolution (e.g., 32).
            If None or <= 0, no change is applied.
    """
    if resolution is None or resolution <= 0:
        return

    try:
        param_x = sim.visionintparam_resolution_x
        param_y = sim.visionintparam_resolution_y
    except Exception:
        print("  LiDAR resolution change not supported by this API build.")
        return

    for handle in lidar_handles.values():
        try:
            sim.setObjectInt32Param(handle, param_x, int(resolution))
            sim.setObjectInt32Param(handle, param_y, int(resolution))
        except Exception as e:
            print(f"  LiDAR resolution set failed: {e}")


def get_lidar_handles(sim, sensor_paths=None):
    """
    Loads all LiDAR sensor handles from CoppeliaSim.

    Looks up each sensor path in the scene and returns a
    dictionary mapping sensor name to its object handle.

    Args:
        sim: CoppeliaSim remote API object.
        sensor_paths (dict[str, str], optional): Mapping of sensor
            name to its scene path. Defaults to LIDAR_SENSOR_PATHS.

    Returns:
        dict[str, int]: Mapping of sensor name to its object handle.
    """
    if sensor_paths is None:
        sensor_paths = LIDAR_SENSOR_PATHS

    handles = {}
    for name, path in sensor_paths.items():
        handles[name] = sim.getObject(path)
    return handles


def get_lidar_min_distance(sim, lidar_handle, max_range=DEFAULT_MAX_RANGE):
    """
    Retrieves depth buffer data from a CoppeliaSim vision sensor,
    unpacks the raw bytes as float32, and returns the smallest
    valid distance reading.

    Args:
        sim: CoppeliaSim remote API object.
        lidar_handle (int): Handle to the vision sensor object.
        max_range (float): Value returned when no valid reading
            is available. Defaults to DEFAULT_MAX_RANGE.

    Returns:
        float: The minimum valid depth reading in metres,
            or max_range if no valid reading is available.
    """
    try:
        result = sim.getVisionSensorDepth(lidar_handle, 0)
        if result is None:
            return max_range

        depth_data, resolution = result[0], result[1]

        if not depth_data or len(depth_data) == 0:
            return max_range

        # depth_data is a raw byte buffer — unpack it as float32
        depth_array = np.frombuffer(depth_data, dtype=np.float32)
        valid = depth_array[depth_array > 0.01]

        if len(valid) == 0:
            return max_range

        return float(np.min(valid))

    except Exception as e:
        print(f'  LiDAR read error: {e}')
        return max_range


def read_lidar_array(sim, lidar_handles):
    """
    Reads all LiDAR sensors and returns distances as a numpy array.

    Returns readings in a fixed order: [Front, Back, Left, Right]
    for consistent use as part of the RL observation vector.

    Args:
        sim: CoppeliaSim remote API object.
        lidar_handles (dict[str, int]): Mapping of sensor name to
            its CoppeliaSim object handle. Must contain keys
            'F', 'B', 'L', 'R'.

    Returns:
        np.ndarray: Shape (4,) array of float32 distances in metres.
    """
    return np.array([
        get_lidar_min_distance(sim, lidar_handles['F']),
        get_lidar_min_distance(sim, lidar_handles['B']),
        get_lidar_min_distance(sim, lidar_handles['L']),
        get_lidar_min_distance(sim, lidar_handles['R']),
    ], dtype=np.float32)


