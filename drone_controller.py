import time
import math

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Connection
client = RemoteAPIClient()
sim = client.require('sim')

# Config Settings
FLIGHT_HEIGHT   = 1.5
REACH_THRESHOLD = 0.5
PAUSE_AT_WP = 0
SPINUP_DELAY = 0

# Dummy Names acting as waypoint positions
WAYPOINT_NAMES = ['/pos1', '/pos2', '/pos3', '/pos4', '/pos5', '/pos6']

SPEED = 0.025

# Handles
drone = sim.getObject('/Quadcopter')
target = sim.getObject('/target')

def get_drone_pos():
    return sim.getObjectPosition(drone, sim.handle_world)

def set_target(x, y, z):
    sim.setObjectPosition(target, sim.handle_world, [x, y, z])

def distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

def load_waypoints():
    waypoints = []
    for name in WAYPOINT_NAMES:
        try:
            handle = sim.getObject(name)
            pos = sim.getObjectPosition(handle, sim.handle_world)
            if pos[2] < 0.5:
                pos[2] = FLIGHT_HEIGHT
            waypoints.append(pos)
            print(f'Loaded waypoint {name}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}')
        except Exception as e:
            print(f'Could not load {name}: {e}')
    return waypoints

def runSimulation():
    print('Loading waypoint(s)...')
    waypoints = load_waypoints()

    if not waypoints:
        print('No waypoints found, aborting.')
        return

    print(f'\n{len(waypoints)} waypoints loaded. Starting simulation...')
    sim.startSimulation()
    time.sleep(SPINUP_DELAY)

    for idx, wp in enumerate(waypoints):
        print(f'\nHeading to waypoint [{idx + 1}/{len(waypoints)}]')

        while True:
            pos = get_drone_pos()
            d = distance(pos, wp)
            print(f'  Distance to waypoint: [{d:.3f} m]', end='\r')

            current_target = sim.getObjectPosition(target, sim.handle_world)
            new_x = current_target[0] + (wp[0] - current_target[0]) * SPEED
            new_y = current_target[1] + (wp[1] - current_target[1]) * SPEED
            new_z = current_target[2] + (wp[2] - current_target[2]) * SPEED
            set_target(new_x, new_y, new_z)

            if d < REACH_THRESHOLD:
                print(f'\n  Reached waypoint [{idx + 1}]')
                time.sleep(PAUSE_AT_WP)
                break

    print('\nAll waypoints reached: Landing...')
    pos = get_drone_pos()
    set_target(pos[0], pos[1], 0.15)
    time.sleep(2)

    print('Simulation Concluded')
    sim.stopSimulation()

runSimulation()