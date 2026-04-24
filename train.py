"""PPO training pipeline for drone obstacle avoidance.

Orchestrates the full training loop:

1. (Optional) Launch one or more headless CoppeliaSim instances.
2. Build a vectorized Gymnasium environment with ``VecNormalize``.
3. Create or resume a PPO model (stable-baselines3).
4. Train for ``TOTAL_TIMESTEPS``, saving the model periodically.
5. Save the final model and normalisation statistics.

Usage::

    python train.py                                           # train from scratch
    python train.py --final-name my_run                       # train with named output
    python train.py --resume --model models/my_run            # resume training
    python train.py --test --model models/my_run              # run one eval episode
"""

import os
import platform
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env import VecNormalize

from drone_environment import DroneAvoidanceEnv


# ── Environment Config ──
ENV_CONFIG = dict(
    max_steps=2000,                    # Max steps before episode is truncated
    speed_scale=0.1,                   # Multiplier from raw action to target-dummy displacement
    collision_distance=0.1,            # LiDAR distance that counts as a crash (m)
    proximity_threshold=0.4,           # Distance below which proximity penalty applies (m)
    boundary_min=-9.0,                 # Min x/y boundary of the flying area (m)
    boundary_max=9.0,                  # Max x/y boundary of the flying area (m)
    min_altitude=0.5,                  # Lowest allowed flight height (m)
    max_altitude=2.5,                  # Highest allowed flight height (m)
    flight_height=1.5,                 # Starting flight height (m)
    ideal_altitude=1.5,                # Ideal cruising altitude -- rewarded for staying near (m)
    exploration_grid_size=0.5,         # Size of grid cells for exploration tracking (m)
    lidar_bins=4,                      # Angular bins per sensor (4 bins x 4 sensors = 16 features)

    # ── Reward weights ──
    survival_reward=0.3,               # Per-step bonus for staying alive
    exploration_reward=20.0,           # One-shot bonus for visiting a new grid cell
    movement_reward=0.0,               # Per-step bonus when horizontal speed > 0.1 m/s (disabled)
    stagnation_start=20,               # Steps in one cell before stagnation penalty accrues
    stagnation_rate=0.3,               # Per-step stagnation penalty accrual rate
    stagnation_cap=8.0,                # Maximum stagnation penalty per step
    proximity_penalty_scale=25.0,      # Multiplier on the proximity penalty
    proximity_penalty_quadratic=True,  # Use quadratic (True) or linear (False) penalty shape
    collision_penalty=60.0,            # Terminal penalty for hitting an obstacle
    out_of_bounds_penalty=50.0,        # Terminal penalty for leaving the flight volume
    hovering_penalty=0.0,              # Per-step penalty when horizontal speed < 0.1 m/s (disabled)
    altitude_bonus=0.5,                # Bonus when inside the altitude soft band
    altitude_soft_band=0.5,            # Half-width of the "free altitude" zone around ideal (m)
    altitude_linear_band=1.0,          # Beyond soft band, linear penalty up to this delta (m)
    altitude_linear_scale=1.0,         # Linear altitude-penalty multiplier
    altitude_quadratic_scale=2.0,      # Quadratic altitude-penalty multiplier beyond linear band
    action_smoothness_scale=0.02,      # Penalty on ||a_t - a_{t-1}||^2 (0 = disabled)
    boundary_warning_distance=0.7,     # Distance from edge where boundary penalty begins (m)
    boundary_penalty_scale=2.0,        # Multiplier on the boundary proximity penalty

    # ── Spawn randomization ──
    randomize_start_pose=True,         # Sample a random (x, y) each reset
    spawn_margin=2.0,                  # Inset from boundaries for spawn area (m)
    spawn_map_path="spawn_map.npy",    # Pre-computed safe spawns (None = rejection sampling)

    # ── Sim / IO ──
    disable_visualization=False,       # Toggle viewport rendering off on reset
    lidar_resolution=16,               # Vision sensor resolution (square, pixels)
)


# ── PPO Hyperparameters ──
PPO_CONFIG = dict(
    learning_rate=3e-4,                # Adam learning rate
    n_steps=256,                       # Steps per rollout before policy update (x NUM_ENVS = total)
    batch_size=64,                     # Minibatch size for each gradient step
    n_epochs=5,                        # SGD passes over each rollout buffer
    gamma=0.995,                       # Discount factor (~200-step effective horizon)
    gae_lambda=0.95,                   # GAE lambda for advantage estimation
    clip_range=0.2,                    # PPO clipping range for policy ratio
    ent_coef=0.005,                    # Entropy bonus coefficient (exploration vs. commitment)
    vf_coef=0.5,                       # Value-function loss weight
    max_grad_norm=0.5,                 # Max gradient norm for clipping
    target_kl=0.04,                    # Early-stop threshold on approx KL divergence
)

# ── Training Config ──
TOTAL_TIMESTEPS = 500_000              # Total training timesteps
SAVE_FREQ = 5_000                      # Overwrite latest model + stats every N steps
HIDDEN_LAYERS = [256, 256]             # Policy and value network architecture
DEVICE = "cpu"                         # Device to train on ("cpu", "cuda", or "auto")
FINAL_MODEL_NAME = ""                  # Optional name (no extension). Example: "drone_ppo_run1"

# ── Multi-instance Config ──
NUM_ENVS = 1                           # Number of parallel sims
BASE_PORT = 23000                      # First ZMQ RPC port
PORT_STRIDE = 2                        # Port increment per instance
HOST = "localhost"

# ── Optional: Launch CoppeliaSim instances from Python ──
# Detects platform and sets paths accordingly.
_project_dir = os.path.dirname(os.path.abspath(__file__))

if platform.system() == "Windows":
    LAUNCH_CONFIG = dict(
        enable=False,
        sim_exe_path="C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/coppeliaSim.exe",
        scene_path=os.path.join(_project_dir, "CoppeliaSim Drone Follower.ttt"),
        headless=False,
        launch_delay=2.0,
    )
else:
    # Linux / Palmetto cluster (inside Apptainer container)
    LAUNCH_CONFIG = dict(
        enable=True,
        sim_exe_path="/opt/coppeliasim/coppeliaSim",
        scene_path=os.path.join(_project_dir, "CoppeliaSim Drone Follower.ttt"),
        headless=True,
        launch_delay=3.0,
        use_xvfb=True,  # Wrap with xvfb-run for headless rendering
    )

# Keep env settings consistent with launcher headless mode
ENV_CONFIG["headless"] = LAUNCH_CONFIG["headless"]

# Track launched simulator processes for cleanup
LAUNCHED_PROCS = []


class TrainingMetricsCallback(BaseCallback):
    """Logs additional per-step metrics to TensorBoard.

    Tracks cumulative collision count, visited grid cells, and
    the minimum LiDAR reading so training progress can be
    monitored beyond the default SB3 scalars.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        """Extracts info dicts and records custom metrics.

        Returns:
            Always ``True`` (never stops training early).
        """
        infos = self.locals.get("infos", [])
        for info in infos:
            if "collision" in info:
                self.episode_count += 1
                self.logger.record("custom/collisions", self.episode_count)
            if "visited_cells" in info:
                self.logger.record("custom/visited_cells", info["visited_cells"])
            if "min_lidar" in info:
                self.logger.record("custom/min_lidar", info["min_lidar"])
        return True


class PeriodicSaveCallback(BaseCallback):
    """Overwrites a "latest" model checkpoint at a fixed interval.

    Also saves the ``VecNormalize`` running statistics so that a
    resumed or evaluated model gets the correct normalisation.

    Args:
        save_freq: Save every this many timesteps.
        save_path: Destination path for the model file.
        stats_path: Destination path for VecNormalize stats
            (optional; skipped if ``None``).
        verbose: Verbosity level.
    """

    def __init__(self, save_freq, save_path, stats_path=None, verbose=0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        self.stats_path = stats_path

    def _on_step(self) -> bool:
        """Saves the model and stats if the interval has elapsed."""
        if self.num_timesteps % self.save_freq == 0:
            try:
                self.model.save(self.save_path)
                if self.stats_path:
                    self.model.get_env().save(self.stats_path)
            except Exception as e:
                if self.verbose:
                    print(f"Periodic save failed: {e}")
        return True


def make_env_fn(rank, port):
    """Returns a zero-arg factory that creates a ``DroneAvoidanceEnv`` on *port*.

    Args:
        rank: Index of this environment (unused, kept for SubprocVecEnv).
        port: ZMQ remote-API port to connect to.

    Returns:
        A callable that produces a configured ``DroneAvoidanceEnv``.
    """
    def _init():
        env = DroneAvoidanceEnv(**ENV_CONFIG, host=HOST, port=port)
        return env
    return _init


def build_vec_env(normalize=True):
    """Builds a vectorized environment, optionally wrapped in VecNormalize.

    ``NUM_ENVS >= 2`` uses ``SubprocVecEnv`` (one process per sim);
    ``NUM_ENVS == 1`` uses ``DummyVecEnv`` (easier to debug).

    Args:
        normalize: If ``True`` (default), wrap with ``VecNormalize``
            for observation and reward normalization.  Pass ``False``
            when you plan to load saved stats via
            ``VecNormalize.load()``.

    Returns:
        A ``VecNormalize`` or ``VecMonitor`` wrapper depending on
        *normalize*.
    """
    if NUM_ENVS == 1:
        vec = DummyVecEnv([make_env_fn(0, BASE_PORT)])
    else:
        ports = [BASE_PORT + i * PORT_STRIDE for i in range(NUM_ENVS)]
        env_fns = [make_env_fn(i, port) for i, port in enumerate(ports)]
        vec = SubprocVecEnv(env_fns)

    vec = VecMonitor(vec)
    if normalize:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec


def launch_coppeliasim_instances():
    """Spawns headless CoppeliaSim processes for each env slot.

    Each instance listens on its own ZMQ port (``BASE_PORT + i *
    PORT_STRIDE``).  Launched processes are tracked in
    ``LAUNCHED_PROCS`` for cleanup on exit.
    """
    if not LAUNCH_CONFIG["enable"]:
        return

    sim_exe = LAUNCH_CONFIG["sim_exe_path"]
    scene = LAUNCH_CONFIG["scene_path"]
    if not sim_exe or not scene:
        raise ValueError("LAUNCH_CONFIG requires sim_exe_path and scene_path when enabled.")

    for i in range(NUM_ENVS):
        port = BASE_PORT + i * PORT_STRIDE
        args = []
        # On Linux, wrap with xvfb-run for headless OpenGL rendering
        if LAUNCH_CONFIG.get("use_xvfb", False):
            args.extend(["xvfb-run", "--auto-servernum", "--server-args=-screen 0 1024x768x24"])
        args.append(sim_exe)
        if LAUNCH_CONFIG["headless"]:
            args.append("-h")
        args.append(f"-GzmqRemoteApi.rpcPort={port}")
        args.extend(["-f", scene])
        proc = subprocess.Popen(args)
        LAUNCHED_PROCS.append(proc)
        time.sleep(LAUNCH_CONFIG["launch_delay"])


def train(resume_path=None):
    """Runs the full PPO training loop.

    Creates the environment, builds the PPO model with the
    configured hyperparameters, and trains for ``TOTAL_TIMESTEPS``.
    Saves the model and VecNormalize stats every ``SAVE_FREQ``
    steps and on completion or Ctrl+C interrupt.

    Args:
        resume_path: Path to a saved model to resume from
            (without ``.zip`` extension).  ``None`` starts fresh.
    """

    os.makedirs("models", exist_ok=True)
    os.makedirs("stats", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Run name — used for model saves, stats, and TensorBoard logs
    run_name = FINAL_MODEL_NAME.strip() or "drone_ppo_final"
    stats_path = f"stats/{run_name}_vecnormalize.pkl"

    # Device
    if DEVICE == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = DEVICE
    print(f"Training on: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Optional: launch sims
    launch_coppeliasim_instances()

    # Environment (with VecNormalize)
    if resume_path and os.path.exists(stats_path):
        # Resume: load saved VecNormalize stats so the running mean/std
        # match what the model was trained with.
        print(f"Loading VecNormalize stats from: {stats_path}")
        env = VecNormalize.load(stats_path, build_vec_env(normalize=False))
    else:
        env = build_vec_env(normalize=True)

    # PPO Model — load existing or create new
    if resume_path:
        print(f"Resuming training from: {resume_path}")
        model = PPO.load(
            resume_path,
            env=env,
            device=device,
            tensorboard_log="./logs/",
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            **PPO_CONFIG,
            verbose=1,
            device=device,
            tensorboard_log="./logs/",
            policy_kwargs=dict(
                net_arch=dict(
                    pi=HIDDEN_LAYERS,
                    vf=HIDDEN_LAYERS,
                ),
                activation_fn=torch.nn.ReLU,
            ),
        )

    print("\n" + "=" * 60)
    print("  Drone Obstacle Avoidance — PPO Training")
    print("=" * 60)
    print(f"  Observation:  {env.observation_space.shape}  (4 lidar + 3 pos + 3 vel)")
    print(f"  Action:       {env.action_space.shape}  (dx, dy, dz)")
    print(f"  Device:       {device}")
    print(f"  Network:      {' → '.join(str(n) for n in HIDDEN_LAYERS)} (ReLU)")
    print(f"  LR:           {PPO_CONFIG['learning_rate']}")
    print(f"  Batch size:   {PPO_CONFIG['batch_size']}")
    print(f"  Gamma:        {PPO_CONFIG['gamma']}")
    print(f"  Max steps:    {ENV_CONFIG['max_steps']} per episode")
    print(f"  Total:        {TOTAL_TIMESTEPS:,} timesteps")
    if resume_path:
        print(f"  Resumed from: {resume_path}")
    print("=" * 60 + "\n")

    # ── Callbacks ──
    metrics_cb = TrainingMetricsCallback()
    latest_path = f"models/{run_name}"
    latest_cb = PeriodicSaveCallback(
        save_freq=SAVE_FREQ,
        save_path=latest_path,
        stats_path=stats_path,
    )

    # ── Train ──
    print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
    print("Monitor with: tensorboard --logdir ./logs/\n")

    def _safe_close_env(env_obj):
        if env_obj is None:
            return
        try:
            env_obj.close()
        except (EOFError, BrokenPipeError):
            pass
        except Exception:
            pass

    shutdown_requested = {"flag": False}

    interrupted = {"flag": False}
    force_exit_timer = {"started": False}

    def _force_exit():
        os._exit(1)

    def _cleanup_and_exit(signum=None, frame=None):
        if shutdown_requested["flag"]:
            return
        shutdown_requested["flag"] = True
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)
        print("\nStopping training and shutting down sims...")
        # Start a 10-second watchdog: if graceful shutdown hangs (e.g.
        # a ZMQ call blocks forever), force-exit the process.
        if not force_exit_timer["started"]:
            force_exit_timer["started"] = True
            threading.Timer(10.0, _force_exit).start()
        for proc in LAUNCHED_PROCS:
            try:
                proc.terminate()
            except Exception:
                pass
        raise KeyboardInterrupt

    # Ctrl+C and Ctrl+Break handling (Windows)
    signal.signal(signal.SIGINT, _cleanup_and_exit)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _cleanup_and_exit)

    try:
        callbacks = [metrics_cb, latest_cb]

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=resume_path is None,
            tb_log_name=run_name,
        )
    except KeyboardInterrupt:
        interrupted["flag"] = True
        print("\nTraining interrupted — saving current model...")
        final_path = f"models/{run_name}"
        try:
            model.save(final_path)
            env.save(stats_path)
            print(f"Model saved to {final_path}.zip")
            print(f"Stats saved to {stats_path}")
        except Exception as e:
            print(f"Save failed: {e}")
    finally:
        if not shutdown_requested["flag"]:
            _safe_close_env(env)
        for proc in LAUNCHED_PROCS:
            try:
                proc.terminate()
            except Exception:
                pass

    if interrupted["flag"]:
        return

    # ── Save ──
    final_path = f"models/{run_name}"
    model.save(final_path)
    env.save(stats_path)
    print(f"Model saved to {final_path}.zip")
    print(f"Stats saved to {stats_path}")

    if not shutdown_requested["flag"]:
        _safe_close_env(env)


def test(model_path="models/drone_ppo_final"):
    """Runs a trained model for one episode with deterministic actions.

    Builds a ``VecNormalize``-wrapped env in eval mode, loads the
    saved normalisation stats alongside the model weights, and
    prints live telemetry every 50 steps.

    Args:
        model_path: Path to the saved model file (without ``.zip``
            extension).
    """
    # Derive stats path from the model name
    run_name = os.path.basename(model_path)
    stats_path = f"stats/{run_name}_vecnormalize.pkl"

    # Build env with VecNormalize in eval mode (don't update stats)
    if os.path.exists(stats_path):
        print(f"Loading VecNormalize stats from: {stats_path}")
        env = VecNormalize.load(stats_path, build_vec_env(normalize=False))
    else:
        print("Warning: no VecNormalize stats found, using fresh normalization")
        env = build_vec_env(normalize=True)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)
    print(f"Loaded model: {model_path}")

    # VecEnv API: obs shape is (1, obs_dim), rewards/dones are arrays
    obs = env.reset()
    total_reward = 0.0
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        total_reward += float(rewards[0])
        step += 1
        info = infos[0]

        if step % 50 == 0:
            # Get unnormalized obs for human-readable telemetry
            raw_obs = env.get_original_obs()[0]
            lidar = raw_obs[:4]
            print(
                f"  Step {step:4d} | "
                f"Reward: {total_reward:7.1f} | "
                f"Cells: {info.get('visited_cells', 0):3d} | "
                f"LiDAR  F:{lidar[0]:.2f}  B:{lidar[1]:.2f}  "
                f"L:{lidar[2]:.2f}  R:{lidar[3]:.2f}"
            )

        if dones[0]:
            print(f"\n  Episode done: {info}")
            print(f"  Total reward:   {total_reward:.1f}")
            print(f"  Steps survived: {step}")
            print(f"  Cells explored: {info.get('visited_cells', 0)}")
            break

    env.close()


if __name__ == "__main__":
    final_name_arg = None
    model_path_arg = None
    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            model_path_arg = sys.argv[i + 1]
        if arg == "--final-name" and i + 1 < len(sys.argv):
            final_name_arg = sys.argv[i + 1]

    if final_name_arg:
        FINAL_MODEL_NAME = final_name_arg

    if "--test" in sys.argv:
        path = model_path_arg or "models/drone_ppo_final"
        test(path)
    elif "--resume" in sys.argv:
        path = model_path_arg or "models/drone_ppo_final"
        train(resume_path=path)
    else:
        train()
