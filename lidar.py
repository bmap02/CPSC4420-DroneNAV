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


def _get_depth_buffer(sim, lidar_handle):
    """
    Reads the raw depth buffer from a vision sensor and returns it
    as a 2D numpy array (rows x cols).  Returns None on failure.
    """
    try:
        result = sim.getVisionSensorDepth(lidar_handle, 0)
        if result is None:
            return None

        depth_data, resolution = result[0], result[1]
        if not depth_data or len(depth_data) == 0:
            return None

        depth_array = np.frombuffer(depth_data, dtype=np.float32)
        rows, cols = resolution[1], resolution[0]
        return depth_array.reshape(rows, cols)

    except Exception as e:
        print(f'  LiDAR read error: {e}')
        return None


def get_lidar_min_distance(sim, lidar_handle, max_range=DEFAULT_MAX_RANGE):
    """
    Returns the single minimum valid depth from a vision sensor.
    Legacy 1-scalar-per-sensor reader.
    """
    buf = _get_depth_buffer(sim, lidar_handle)
    if buf is None:
        return max_range

    valid = buf[buf > 0.01]
    if len(valid) == 0:
        return max_range

    return float(np.min(valid))


def get_lidar_binned_distances(sim, lidar_handle, num_bins=4,
                                max_range=DEFAULT_MAX_RANGE):
    """
    Returns per-column-bin minimum depths from a vision sensor.

    Splits the depth buffer into *num_bins* vertical column slices
    (left-to-right across the sensor's horizontal FOV) and returns
    the minimum valid depth in each slice.  4 bins on a 90-degree
    sensor gives ~22.5 degrees per bin.

    Args:
        sim: CoppeliaSim remote API object.
        lidar_handle (int): Vision sensor handle.
        num_bins (int): Number of horizontal angular bins.
        max_range (float): Value for bins with no valid reading.

    Returns:
        np.ndarray: Shape (num_bins,) float32 array of distances.
    """
    buf = _get_depth_buffer(sim, lidar_handle)
    if buf is None:
        return np.full(num_bins, max_range, dtype=np.float32)

    cols = buf.shape[1]
    bin_width = max(1, cols // num_bins)
    result = np.full(num_bins, max_range, dtype=np.float32)

    for i in range(num_bins):
        start = i * bin_width
        end = start + bin_width if i < num_bins - 1 else cols
        col_slice = buf[:, start:end]
        valid = col_slice[col_slice > 0.01]
        if len(valid) > 0:
            result[i] = float(np.min(valid))

    return result


def read_lidar_array(sim, lidar_handles, num_bins=1):
    """
    Reads all 4 LiDAR sensors and returns distances as a numpy array.

    With num_bins=1 (default), returns shape (4,) — one min per sensor
    (legacy behaviour).

    With num_bins=N, returns shape (4*N,) — N angular bins per sensor
    in order [F_bin0..F_binN-1, B_..., L_..., R_...].

    Args:
        sim: CoppeliaSim remote API object.
        lidar_handles (dict[str, int]): Must contain 'F', 'B', 'L', 'R'.
        num_bins (int): Angular bins per sensor. 1 = legacy single-min.

    Returns:
        np.ndarray: Shape (4*num_bins,) float32 distances in metres.
    """
    if num_bins <= 1:
        return np.array([
            get_lidar_min_distance(sim, lidar_handles['F']),
            get_lidar_min_distance(sim, lidar_handles['B']),
            get_lidar_min_distance(sim, lidar_handles['L']),
            get_lidar_min_distance(sim, lidar_handles['R']),
        ], dtype=np.float32)

    return np.concatenate([
        get_lidar_binned_distances(sim, lidar_handles['F'], num_bins),
        get_lidar_binned_distances(sim, lidar_handles['B'], num_bins),
        get_lidar_binned_distances(sim, lidar_handles['L'], num_bins),
        get_lidar_binned_distances(sim, lidar_handles['R'], num_bins),
    ])


