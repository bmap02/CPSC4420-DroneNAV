"""Generate a safe-spawn map by sweeping the drone across the scene.

The script teleports a *static* (non-dynamic) drone to every grid
position at ``FLIGHT_HEIGHT`` and records which positions have clear
LiDAR readings.  Positions where ``min(lidar) > SAFE_THRESHOLD`` are
saved to ``spawn_map.npy`` as an Nx2 array of (x, y) coordinates.

Strategy:
    1. Disable the drone's flight script so the PID controller does
       not fight the teleport.
    2. Set the drone (and its respondable body) to *static* so
       physics are frozen -- no gravity, no collisions, no tipping.
    3. Teleport to each grid cell, step the sim once to update the
       vision sensors, and read the LiDAR.
    4. Restore dynamic mode and re-enable the flight script when done.

SAFE_THRESHOLD should be >= ``proximity_threshold`` in ``train.py``
``ENV_CONFIG`` so that no spawn puts the drone inside its own penalty
zone on the very first step.

Usage::

    1. Open CoppeliaSim with the scene (do NOT start the simulation).
    2. Run:  python generate_spawn_map.py
    3. Output: spawn_map.npy + spawn_map_preview.txt

Requires one CoppeliaSim instance running on port 23000.
"""

import time

import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from lidar import get_lidar_handles, read_lidar_array

# ── Config ──
HOST = "localhost"
PORT = 23000
GRID_STEP = 0.5          # Metres between sweep positions
BOUNDARY_MIN = -8.0      # Inset from the actual flight bounds (+/-9) so
BOUNDARY_MAX = 8.0       # spawns never land in the boundary warning zone
FLIGHT_HEIGHT = 1.5
# Minimum LiDAR reading to consider a position safe.
# Cross-ref: should be >= proximity_threshold in train.py ENV_CONFIG
# so newly-spawned drones do not immediately incur a penalty.
SAFE_THRESHOLD = 0.25
OUTPUT_FILE = "spawn_map.npy"
PREVIEW_FILE = "spawn_map_preview.txt"


def main():
    """Runs the full sweep and writes ``spawn_map.npy``."""
    print("Connecting to CoppeliaSim...")
    client = RemoteAPIClient(host=HOST, port=PORT)
    sim = client.require("sim")

    drone = sim.getObject("/Quadcopter")
    target = sim.getObject("/target")
    lidar_handles = get_lidar_handles(sim)
    flight_script = sim.getScript(sim.scripttype_childscript, drone, "")

    # Build the grid
    xs = np.arange(BOUNDARY_MIN, BOUNDARY_MAX + GRID_STEP, GRID_STEP)
    ys = np.arange(BOUNDARY_MIN, BOUNDARY_MAX + GRID_STEP, GRID_STEP)
    total = len(xs) * len(ys)
    print(f"Grid: {len(xs)} x {len(ys)} = {total} positions")
    print(f"Step: {GRID_STEP}m, bounds: [{BOUNDARY_MIN}, {BOUNDARY_MAX}]")

    # Stop any running sim, then start fresh
    try:
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.05)
    except Exception:
        pass

    sim.startSimulation()
    time.sleep(0.5)

    # Disable the flight script so the PID controller does not interfere
    # with teleportation.  Making the drone static freezes it in place --
    # no gravity, no physics, no tipping -- turning it into a pure sensor
    # platform that we teleport around the grid.
    sim.setScriptInt32Param(flight_script, sim.scriptintparam_enabled, 0)
    sim.setObjectInt32Param(drone, sim.shapeintparam_static, 1)
    # Also freeze the respondable body (collision hull)
    try:
        respondable = sim.getObject("/Quadcopter/respondable")
        sim.setObjectInt32Param(respondable, sim.shapeintparam_static, 1)
    except Exception:
        pass
    print("Flight script disabled, drone set to static for sweep")

    # Sweep every grid position
    safe_map = np.zeros((len(ys), len(xs)), dtype=np.uint8)
    checked = 0

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            sim.setObjectPosition(drone, sim.handle_world,
                                  [float(x), float(y), FLIGHT_HEIGHT])
            # Single step to update sensors
            sim.step()

            lidar = read_lidar_array(sim, lidar_handles)
            min_dist = float(np.min(lidar))

            if min_dist > SAFE_THRESHOLD:
                safe_map[iy, ix] = 1

            checked += 1
            if checked % 100 == 0 or checked == total:
                safe_count = int(np.sum(safe_map))
                print(f"  {checked}/{total} checked, {safe_count} safe so far "
                      f"({100*safe_count/checked:.0f}%)")

    # Restore drone to dynamic and re-enable flight script
    sim.setObjectInt32Param(drone, sim.shapeintparam_static, 0)
    try:
        sim.setObjectInt32Param(respondable, sim.shapeintparam_static, 0)
    except Exception:
        pass
    sim.setScriptInt32Param(flight_script, sim.scriptintparam_enabled, 1)
    sim.stopSimulation()
    print("Drone restored to dynamic, flight script re-enabled, sim stopped")

    # Collect safe positions as a simple Nx2 array of (x, y) coordinates
    safe_positions = []
    for iy in range(len(ys)):
        for ix in range(len(xs)):
            if safe_map[iy, ix]:
                safe_positions.append([xs[ix], ys[iy]])

    safe_positions = np.array(safe_positions, dtype=np.float32)
    print(f"\nDone: {len(safe_positions)}/{total} safe positions "
          f"({100*len(safe_positions)/total:.0f}%)")

    # Save just the safe (x, y) coordinates
    np.save(OUTPUT_FILE, safe_positions)
    print(f"Saved: {OUTPUT_FILE}")

    # Save a human-readable preview
    with open(PREVIEW_FILE, "w") as f:
        f.write(f"Spawn map: {len(xs)}x{len(ys)} grid, step={GRID_STEP}m\n")
        f.write(f"Bounds: [{BOUNDARY_MIN}, {BOUNDARY_MAX}]\n")
        f.write(f"Safe: {len(safe_positions)}/{total} "
                f"({100*len(safe_positions)/total:.0f}%)\n\n")
        # Print map with . = safe, X = obstacle, oriented so +y is up
        for iy in range(len(ys) - 1, -1, -1):
            row = ""
            for ix in range(len(xs)):
                row += ". " if safe_map[iy, ix] else "X "
            f.write(f"y={ys[iy]:+6.1f} |{row}\n")
        f.write(f"         ")
        for ix in range(0, len(xs), 4):
            f.write(f"x={xs[ix]:+.0f}    ")
        f.write("\n")
    print(f"Saved: {PREVIEW_FILE}")


if __name__ == "__main__":
    main()
