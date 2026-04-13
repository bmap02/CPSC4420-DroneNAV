"""
Gymnasium environment for CoppeliaSim drone obstacle avoidance.

The drone's goal is to fly freely, explore the environment, and
avoid obstacles (trees, people, buildings) for as long as possible.
No waypoints or destinations — purely learning-based navigation.

Follows the standard Gymnasium API:
    env.reset()  → (observation, info)
    env.step()   → (observation, reward, terminated, truncated, info)

Observation (10D):
    [0:4]  — 4 LiDAR distances (front, back, left, right)
    [4:7]  — drone position (x, y, z)
    [7:10] — drone velocity (vx, vy, vz)

Action (3D):
    Velocity adjustments (dx, dy, dz) in [-1, 1], clipped and scaled
    by speed_scale before being sent to the target dummy. No hand-coded
    nudging — the policy is the sole source of control.

Reward (all weights configurable via __init__ kwargs / train.py ENV_CONFIG):
    + survival_reward          per step survived
    + exploration_reward       one-shot bonus for visiting a new grid cell
    + movement_reward          when horizontal speed > 0.1 m/s
    + altitude_bonus           when inside the altitude soft band
    - proximity penalty        linear or quadratic in (proximity_threshold - min_lidar)
    - stagnation penalty       after camping in one cell too long
    - altitude penalty         linear then quadratic outside the soft band
    - hovering penalty         when horizontal speed < 0.1 m/s
    - action smoothness        scale * ||a_t - a_{t-1}||^2 (0 disables)
    - collision_penalty        terminal, episode ends
    - out_of_bounds_penalty    terminal, episode ends
"""

import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from lidar import get_lidar_handles, read_lidar_array, set_lidar_resolution
from navigation import get_drone_pos_array, get_drone_velocity, set_target


class DroneAvoidanceEnv(gym.Env):
    """
    CoppeliaSim drone environment for obstacle avoidance.

    Implements the standard Gymnasium Env interface so it can be
    used with any compatible RL library (stable-baselines3, etc).

    The agent controls the drone's velocity in 3D space. Altitude
    is clamped and penalized to discourage cheating by flying above
    obstacles.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        *,
        # ── Episode / control (required — supplied by train.py ENV_CONFIG) ──
        max_steps,                         # Max steps before episode ends
        speed_scale,                       # Scales agent action into movement
        collision_distance,                # LiDAR distance that counts as a crash (m)
        proximity_threshold,               # Distance below which proximity penalty applies (m)
        boundary_min,                      # Min x/y boundary of the flying area (m)
        boundary_max,                      # Max x/y boundary of the flying area (m)
        min_altitude,                      # Lowest allowed flight height (m)
        max_altitude,                      # Highest allowed flight height (m)
        flight_height,                     # Starting flight height (m)
        ideal_altitude,                    # Ideal cruising altitude (m) — reward for staying near
        exploration_grid_size,             # Size of grid cells for exploration tracking (m)
        # ── Reward weights (required — single source of truth is train.py) ──
        survival_reward,                   # Per-step bonus for staying alive
        exploration_reward,                # One-shot bonus for visiting a new grid cell
        movement_reward,                   # Per-step bonus when horizontal speed > 0.1 m/s
        stagnation_start,                  # Steps in one cell before stagnation kicks in
        stagnation_rate,                   # Per-step penalty accrual while stagnating
        stagnation_cap,                    # Max stagnation penalty per step
        proximity_penalty_scale,           # Multiplier on the proximity penalty
        proximity_penalty_quadratic,       # If True, use ((1-d/T)^2) * scale; else linear
        collision_penalty,                 # Terminal penalty for hitting an obstacle
        out_of_bounds_penalty,             # Terminal penalty for leaving the flight volume
        hovering_penalty,                  # Per-step penalty when speed < 0.1 m/s
        altitude_bonus,                    # Bonus when within altitude_soft_band of ideal
        altitude_soft_band,                # Half-width of the "free altitude" band (m)
        altitude_linear_band,              # Beyond this Δ, switch to quadratic penalty (m)
        altitude_linear_scale,             # Linear penalty multiplier inside linear band
        altitude_quadratic_scale,          # Quadratic penalty multiplier outside linear band
        action_smoothness_scale,           # Penalty on ||a_t - a_{t-1}||^2 (0 = disabled)
        # ── Spawn randomization ──
        randomize_start_pose,              # If True, sample (x, y) each reset
        spawn_margin,                      # Inset from boundary_min/max for spawn area (m)
        # ── Sim / IO (required — supplied by train.py) ──
        disable_visualization,             # Toggle visualization off on reset
        lidar_resolution,                  # Vision sensor resolution (square). None to skip
        headless,                          # Skip display toggles when running headless
        host,                              # Remote API host
        port,                              # Remote API port (unique per sim instance)
        # ── Debug / optional (have defaults because train.py doesn't pass them) ──
        render_mode=None,                  # Gymnasium render mode
        profile_every=0,                   # Print timing stats every N steps (0 disables)
        cntport=None,                      # Remote API control port (optional)
    ):
        """
        Initializes the drone environment.

        Sets up the observation and action spaces, stores all
        configuration parameters, and prepares internal state.
        Does NOT connect to CoppeliaSim yet — that happens
        on the first call to reset().

        Args:
            max_steps (int): Max steps before episode is truncated.
            speed_scale (float): Multiplier applied to actions.
            collision_distance (float): LiDAR reading below this
                triggers a crash and ends the episode.
            proximity_threshold (float): LiDAR reading below which the
                proximity penalty starts accruing.
            boundary_min (float): Min x/y world coordinate.
            boundary_max (float): Max x/y world coordinate.
            min_altitude (float): Minimum allowed z position.
            max_altitude (float): Maximum allowed z position.
            flight_height (float): Starting altitude on reset.
            ideal_altitude (float): Ideal cruising altitude. The
                agent is rewarded for staying near this height
                and penalized for drifting away.
            exploration_grid_size (float): Grid cell size for
                tracking which areas the drone has visited.
            render_mode (str, optional): Gymnasium render mode.
        """
        super().__init__()

        # ── Config ──
        self.max_steps = max_steps
        self.speed_scale = speed_scale
        self.collision_distance = collision_distance
        self.proximity_threshold = proximity_threshold
        self.boundary_min = boundary_min
        self.boundary_max = boundary_max
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.flight_height = flight_height
        self.ideal_altitude = ideal_altitude
        self.exploration_grid_size = exploration_grid_size

        # Reward weights
        self.survival_reward = survival_reward
        self.exploration_reward = exploration_reward
        self.movement_reward = movement_reward
        self.stagnation_start = stagnation_start
        self.stagnation_rate = stagnation_rate
        self.stagnation_cap = stagnation_cap
        self.proximity_penalty_scale = proximity_penalty_scale
        self.proximity_penalty_quadratic = proximity_penalty_quadratic
        self.collision_penalty = collision_penalty
        self.out_of_bounds_penalty = out_of_bounds_penalty
        self.hovering_penalty = hovering_penalty
        self.altitude_bonus = altitude_bonus
        self.altitude_soft_band = altitude_soft_band
        self.altitude_linear_band = altitude_linear_band
        self.altitude_linear_scale = altitude_linear_scale
        self.altitude_quadratic_scale = altitude_quadratic_scale
        self.action_smoothness_scale = action_smoothness_scale

        # Spawn randomization
        self.randomize_start_pose = randomize_start_pose
        self.spawn_margin = spawn_margin

        # Sim / IO
        self.render_mode = render_mode
        self.profile_every = profile_every
        self.disable_visualization = disable_visualization
        self.lidar_resolution = lidar_resolution
        self.headless = headless
        self.host = host
        self.port = port
        self.cntport = cntport

        # Observation space
        # 4 lidar readings + 3 position + 3 velocity = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # Action space
        # 3D velocity adjustment (x, y, z) each in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # CoppeliaSim handles (set on first reset)
        self.client = None
        self.sim = None
        self.drone = None
        self.target = None
        self.lidar_handles = {}

        # ── Episode state ──
        self.step_count = 0
        self.visited_cells = set()
        self.steps_in_current_cell = 0
        self.last_cell = None
        self.prev_action = np.zeros(3, dtype=np.float32)
        self._connected = False

        # Profiling accumulators (seconds)
        self._profile = {
            "steps": 0,
            "target_move": 0.0,
            "sim_step": 0.0,
            "observation": 0.0,
            "reward": 0.0,
            "total": 0.0,
        }

    #  Connection
    def _connect(self):
        """
        Establishes connection to CoppeliaSim.

        Grabs handles for the drone, target dummy, and all four
        LiDAR sensors. Skips if already connected.
        """
        if self._connected:
            return

        self.client = RemoteAPIClient(host=self.host, port=self.port, cntport=self.cntport)
        self.sim = self.client.require('sim')

        self.drone = self.sim.getObject('/Quadcopter')
        self.target = self.sim.getObject('/target')
        self.lidar_handles = get_lidar_handles(self.sim)

        self._connected = True

    def _apply_sim_settings(self):
        """
        Applies sim settings that tend to reset on simulation start.
        """
        if self.disable_visualization and not self.headless:
            try:
                self.sim.setBoolParam(self.sim.boolparam_display_enabled, False)
            except Exception as e:
                print(f"  Visualization toggle failed: {e}")

        if self.lidar_resolution:
            set_lidar_resolution(self.sim, self.lidar_handles, self.lidar_resolution)

    #  Observation
    def _get_observation(self):
        """
        Builds the 10D observation vector from the current sim state.

        Reads all four LiDAR sensors, the drone's world position,
        and its linear velocity, then concatenates them into a
        single flat array.

        Returns:
            np.ndarray: Shape (10,) float32 observation.
                [0:4]  — LiDAR distances (F, B, L, R) in metres
                [4:7]  — Drone position (x, y, z) in metres
                [7:10] — Drone velocity (vx, vy, vz) in m/s
        """
        lidar = read_lidar_array(self.sim, self.lidar_handles)
        pos = get_drone_pos_array(self.sim, self.drone)
        vel = get_drone_velocity(self.sim, self.drone)

        return np.concatenate([lidar, pos, vel])


    #  Exploration tracking
    def _get_grid_cell(self, pos):
        """
        Converts a world position to a discrete grid cell.

        Used to track which areas the drone has visited so the
        exploration reward can encourage covering new ground.

        Args:
            pos (np.ndarray): ``[x, y, z]`` position in metres.

        Returns:
            tuple[int, int, int]: Grid cell indices (gx, gy, gz).
        """
        gx = int(pos[0] / self.exploration_grid_size)
        gy = int(pos[1] / self.exploration_grid_size)
        gz = int(pos[2] / self.exploration_grid_size)
        return (gx, gy, gz)


    #  Reward
    def _compute_reward(self, obs, action):
        """
        Computes the reward for the current step.

        All reward weights come from ``self`` attributes set in ``__init__``,
        so they can be tuned from ``train.py`` without editing this file.
        The reward signal is a combination of:
            + survival bonus (awarded every step)
            + exploration bonus (new grid cell visited, one-shot per cell)
            + movement bonus (speed above 0.1 m/s)
            + altitude bonus (within soft band of ideal altitude)
            - proximity penalty (scales with closeness to obstacles)
            - stagnation penalty (camping in one cell)
            - altitude penalty (linear then quadratic outside soft band)
            - hovering penalty (speed below 0.1 m/s)
            - action smoothness penalty (||a_t - a_{t-1}||^2)
            - collision (episode terminates)
            - out of bounds (episode terminates)

        Args:
            obs (np.ndarray): Current 10D observation vector.
            action (np.ndarray): The (clipped) raw action the policy
                emitted this step, used for the smoothness penalty.

        Returns:
            tuple[float, bool, dict]:
                reward (float): Scalar reward value.
                terminated (bool): Whether the episode ended
                    due to collision or out of bounds.
                info (dict): Additional diagnostics about what
                    triggered the reward components.
        """
        lidar = obs[:4]
        pos = obs[4:7]
        vel = obs[7:10]

        reward = 0.0
        terminated = False
        info = {}

        # Survival reward
        reward += self.survival_reward

        # Exploration reward — one-shot per cell
        cell = self._get_grid_cell(pos)
        if cell not in self.visited_cells:
            self.visited_cells.add(cell)
            reward += self.exploration_reward
            info['new_cell'] = True

        # Movement bonus — reward any horizontal motion
        speed = np.linalg.norm(vel[:2])
        if speed > 0.1:
            reward += self.movement_reward

        # Stagnation penalty — punish camping in one spot
        if cell == self.last_cell:
            self.steps_in_current_cell += 1
        else:
            self.steps_in_current_cell = 0
            self.last_cell = cell

        if self.steps_in_current_cell > self.stagnation_start:
            stagnation_penalty = min(
                (self.steps_in_current_cell - self.stagnation_start) * self.stagnation_rate,
                self.stagnation_cap,
            )
            reward -= stagnation_penalty

        # Obstacle proximity penalty (linear or quadratic shape)
        min_lidar = np.min(lidar)
        if min_lidar < self.proximity_threshold:
            closeness = (self.proximity_threshold - min_lidar) / self.proximity_threshold
            if self.proximity_penalty_quadratic:
                proximity_penalty = (closeness ** 2) * self.proximity_penalty_scale
            else:
                # Legacy linear shape kept for behaviour preservation
                proximity_penalty = (self.proximity_threshold - min_lidar) * self.proximity_penalty_scale
            reward -= proximity_penalty

        # Collision
        if min_lidar < self.collision_distance:
            reward -= self.collision_penalty
            terminated = True
            info['collision'] = True

        # Out of bounds
        if (
            pos[0] < self.boundary_min
            or pos[0] > self.boundary_max
            or pos[1] < self.boundary_min
            or pos[1] > self.boundary_max
            or pos[2] < self.min_altitude
            or pos[2] > self.max_altitude
        ):
            reward -= self.out_of_bounds_penalty
            terminated = True
            info['out_of_bounds'] = True

        # Hovering penalty
        if speed < 0.1:
            reward -= self.hovering_penalty

        # Altitude reward/penalty — keep drone near ideal height
        altitude_diff = abs(pos[2] - self.ideal_altitude)
        if altitude_diff < self.altitude_soft_band:
            reward += self.altitude_bonus
        elif altitude_diff < self.altitude_linear_band:
            reward -= altitude_diff * self.altitude_linear_scale
        else:
            # Squared penalty kicks in hard above the linear band
            reward -= (altitude_diff ** 2) * self.altitude_quadratic_scale

        # Action smoothness penalty — discourage twitchy control
        if self.action_smoothness_scale > 0.0:
            action_delta = action - self.prev_action
            reward -= self.action_smoothness_scale * float(np.sum(action_delta ** 2))

        # Max steps
        if self.step_count >= self.max_steps:
            info['timeout'] = True

        return reward, terminated, info


    #  Gymnasium API: reset
    def reset(self, seed=None, options=None):
        """
        Resets the environment for a new episode.

        Follows the Gymnasium API:
            observation, info = env.reset()

        Stops any running simulation, restarts it, clears the
        step counter and visited cells, and places the drone
        at its starting flight height.

        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional reset options.

        Returns:
            tuple[np.ndarray, dict]:
                observation (np.ndarray): Initial 10D observation.
                info (dict): Reset diagnostics.
        """
        # Initialize the RNG (Gymnasium requirement)
        super().reset(seed=seed)

        # Connect to CoppeliaSim on first call
        self._connect()

        # Stop any running simulation
        try:
            self.sim.stopSimulation()
            while self.sim.getSimulationState() != self.sim.simulation_stopped:
                time.sleep(0.1)
        except Exception:
            pass

        # Apply sim settings before starting
        self._apply_sim_settings()

        # Start a fresh simulation
        self.sim.startSimulation()
        time.sleep(0.5)

        # Reset episode state
        self.step_count = 0
        self.visited_cells = set()
        self.steps_in_current_cell = 0
        self.last_cell = None
        self.prev_action = np.zeros(3, dtype=np.float32)

        # Place drone and target at the starting position
        if self.randomize_start_pose:
            lo = self.boundary_min + self.spawn_margin
            hi = self.boundary_max - self.spawn_margin
            z = self.flight_height
            safe_threshold = max(self.collision_distance * 3, 0.3)

            # Rejection sampling: keep trying until we find a spawn that
            # isn't inside or right next to an obstacle.
            for attempt in range(20):
                x = float(self.np_random.uniform(lo, hi))
                y = float(self.np_random.uniform(lo, hi))
                self.sim.setObjectPosition(
                    self.drone, self.sim.handle_world, [x, y, z]
                )
                set_target(self.sim, self.target, x, y, z)
                # Step a few times so sensors + flight controller update
                for _ in range(10):
                    self.sim.step()
                lidar = read_lidar_array(self.sim, self.lidar_handles)
                if np.min(lidar) > safe_threshold:
                    break  # Safe spawn
        else:
            pos = get_drone_pos_array(self.sim, self.drone)
            set_target(self.sim, self.target, pos[0], pos[1], self.flight_height)

        # Build first observation
        observation = self._get_observation()
        info = {'visited_cells': 0, 'start_pos': [x, y, z] if self.randomize_start_pose else None}

        return observation, info


    #  Gymnasium API: step
    def step(self, action):
        """
        Executes one environment step.

        Follows the Gymnasium API:
            observation, reward, terminated, truncated, info = env.step(action)

        Clips the agent's action to [-1, 1], scales it by speed_scale,
        and moves the drone's target position accordingly. Then steps
        the simulation forward and computes the new observation and
        reward.

        Args:
            action (np.ndarray): Shape (3,) array with values
                in [-1, 1] representing velocity adjustments
                in (x, y, z).

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]:
                observation (np.ndarray): New 10D observation.
                reward (float): Reward for this step.
                terminated (bool): True if episode ended due to
                    collision or going out of bounds.
                truncated (bool): True if episode was cut short
                    because max_steps was reached.
                info (dict): Step diagnostics including step
                    count, visited cells, and min LiDAR reading.
        """
        self.step_count += 1

        do_profile = self.profile_every and self.profile_every > 0
        if do_profile:
            t_total_start = time.perf_counter()

        # Clip raw action and scale — policy output goes straight to the sim,
        # no hand-coded nudging.
        action = np.clip(action, -1.0, 1.0)
        scaled_action = action * self.speed_scale

        # Get current target position and apply the action
        current_target = self.sim.getObjectPosition(
            self.target, self.sim.handle_world
        )
        new_x = current_target[0] + scaled_action[0]
        new_y = current_target[1] + scaled_action[1]
        new_z = current_target[2] + scaled_action[2]

        # Clamp altitude to allowed range
        new_z = max(self.min_altitude, min(self.max_altitude, new_z))

        if do_profile:
            t0 = time.perf_counter()
        set_target(self.sim, self.target, new_x, new_y, new_z)
        if do_profile:
            self._profile["target_move"] += time.perf_counter() - t0

        # Advance the simulation one step
        if do_profile:
            t0 = time.perf_counter()
        self.sim.step()
        if do_profile:
            self._profile["sim_step"] += time.perf_counter() - t0

        # Build observation and compute reward
        if do_profile:
            t0 = time.perf_counter()
        observation = self._get_observation()
        if do_profile:
            self._profile["observation"] += time.perf_counter() - t0

        if do_profile:
            t0 = time.perf_counter()
        reward, terminated, info = self._compute_reward(observation, action)
        if do_profile:
            self._profile["reward"] += time.perf_counter() - t0
        truncated = self.step_count >= self.max_steps

        # Remember this step's action for next step's smoothness penalty
        self.prev_action = action.astype(np.float32, copy=True)

        # Attach step diagnostics
        info['step'] = self.step_count
        info['visited_cells'] = len(self.visited_cells)
        info['min_lidar'] = float(np.min(observation[:4]))

        if do_profile:
            self._profile["steps"] += 1
            self._profile["total"] += time.perf_counter() - t_total_start

            if self._profile["steps"] % self.profile_every == 0:
                steps = self._profile["steps"]
                def ms(key):
                    return (self._profile[key] / steps) * 1000.0

                print(
                    "Timing avg (ms/step) | "
                    f"total {ms('total'):.2f} | "
                    f"target {ms('target_move'):.2f} | "
                    f"sim.step {ms('sim_step'):.2f} | "
                    f"obs {ms('observation'):.2f} | "
                    f"reward {ms('reward'):.2f}"
                )

        return observation, reward, terminated, truncated, info

    # ──────────────────────────────────────────────
    #  Gymnasium API: close
    # ──────────────────────────────────────────────

    def close(self):
        """
        Stops the simulation and cleans up.

        Safe to call even if the simulation is not running.
        Should be called when finished with the environment.
        """
        if self._connected:
            try:
                self.sim.stopSimulation()
            except Exception:
                pass
