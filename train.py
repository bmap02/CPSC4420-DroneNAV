"""
PPO training script for drone obstacle avoidance.

The drone learns to fly freely through the environment, exploring
while avoiding obstacles using LiDAR sensor input.

Requirements:
    pip install stable-baselines3 gymnasium numpy torch

Usage:
    Train:   python train.py
    Resume:  python train.py --resume --model models/checkpoints/drone_ppo_10000_steps
    Test:    python train.py --test
    Custom:  python train.py --test --model models/checkpoints/drone_ppo_50000_steps
"""

import os
import sys
import platform
import numpy as np
import subprocess
import time
import signal
import threading
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env import VecNormalize

from drone_environment import DroneAvoidanceEnv



#  Environment Config
ENV_CONFIG = dict(
    max_steps=2000,                    # Max steps before episode ends
    speed_scale=0.1,                   # Scales agent action into movement
    collision_distance=0.1,            # LiDAR distance that counts as a crash (m)
    proximity_threshold=0.4,           # WAS 0.25 — bumped for longer push-back gradient
    boundary_min=-9.0,                 # Min x/y boundary of the flying area (m)
    boundary_max=9.0,                  # Max x/y boundary of the flying area (m)
    min_altitude=0.3,                  # Lowest allowed flight height (m)
    max_altitude=3.0,                  # Highest allowed flight height (m)
    flight_height=1.5,                 # Starting flight height (m)
    ideal_altitude=1.5,                # Ideal cruising altitude — rewarded for staying near (m)
    exploration_grid_size=0.5,         # Size of grid cells for exploration tracking (m)
    lidar_bins=4,                      # Angular bins per sensor (4 bins × 4 sensors = 16 features)

    # ── Reward weights (Step 1 rebalance) ───────────────────────────────
    # Goal: break the "big perimeter circle" attractor by removing the
    # per-step movement bonus, sharpening the proximity penalty to a
    # quadratic near-collision shape, softening the altitude band so it
    # doesn't drown out coverage signal, and adding an action-smoothness
    # penalty to suppress twitchy control. See plan §11 Step 1.
    survival_reward=0.3,               # Per-step bonus for staying alive
    exploration_reward=20.0,           # One-shot bonus for visiting a new grid cell
    movement_reward=0.0,               # WAS 4.0 — deleted to break big-circle attractor
    stagnation_start=20,               # Steps in one cell before stagnation accrues
    stagnation_rate=0.3,               # Per-step stagnation penalty accrual
    stagnation_cap=8.0,                # Max stagnation penalty per step
    proximity_penalty_scale=25.0,      # WAS 3.5 — stronger, re-tuned for 0.25m threshold
    proximity_penalty_quadratic=True,  # WAS False — quadratic shape, dominates near d=0
    collision_penalty=60.0,            # Terminal penalty for hitting an obstacle
    out_of_bounds_penalty=50.0,        # Terminal penalty for leaving the flight volume
    hovering_penalty=0.0,              # WAS 1.0 — deleted alongside movement bonus
    altitude_bonus=0.5,                # Bonus inside the altitude soft band
    altitude_soft_band=0.5,            # WAS 0.3 — widened so altitude isn't the main gradient
    altitude_linear_band=1.0,          # Linear-region half-width (m)
    altitude_linear_scale=0.3,         # WAS 1.5 — softened linear slope
    altitude_quadratic_scale=0.5,      # WAS 3.0 — softened quadratic slope
    action_smoothness_scale=0.02,      # NEW — penalty on ||a_t - a_{t-1}||^2
    boundary_warning_distance=0.7,     # NEW — penalty ramps up within 0.7m of any edge
    boundary_penalty_scale=2.0,        # NEW — multiplier on boundary proximity penalty
    # ────────────────────────────────────────────────────────────────────

    # ── Spawn randomization (Step 3) ────────────────────────────────────
    # Breaks the fixed-start attractor by spawning the drone at a random
    # (x, y) each episode. Spawn area is boundary ± margin.
    randomize_start_pose=True,         # NEW — random spawn each reset
    spawn_margin=2.0,                  # Inset from boundaries (m), spawn in [-7, 7]
    spawn_map_path="spawn_map.npy",   # Pre-computed safe spawns (None = rejection sampling)
    # ────────────────────────────────────────────────────────────────────

    disable_visualization=True,        # Toggle visualization off on reset
    lidar_resolution=16,               # Vision sensor resolution (square)
)


#  PPO parameters
PPO_CONFIG = dict(
    learning_rate=3e-4,                # WAS 5e-4 — lowered to fix high approx_kl with VecNormalize
    n_steps=256,                      # Steps per rollout before policy update (× NUM_ENVS = total)
    batch_size=64,                     # Minibatch size for each gradient step
    n_epochs=10,                       # Number of PPO update epochs per rollout
    gamma=0.995,                       # WAS 0.98 — longer planning horizon (~200 steps / 10s at 50ms dt)
    gae_lambda=0.95,                   # GAE lambda for advantage estimation
    clip_range=0.2,                    # PPO clipping range for policy updates
    ent_coef=0.01,                     # Entropy coefficient (encourages exploration)
    vf_coef=0.5,                       # Value function loss weight
    max_grad_norm=0.5,                 # Max gradient norm for clipping
    target_kl=0.025,                   # Stop update early if approx_kl exceeds this
)

#  Evaluation Config
EVAL_CONFIG = dict(
    enabled=False,                      # Enable evaluation runs
    eval_freq=5000,                    # Evaluate every N training timesteps
    n_eval_episodes=1,                 # Number of eval episodes per check
    seed=123,                          # Base seed for eval episodes
    use_separate_env=True,             # Use a dedicated eval sim instance
)

#  Training Config
TOTAL_TIMESTEPS = 500_000              # Total training timesteps
CHECKPOINT_FREQ = 10_000               # Save model every N steps
LATEST_SAVE_FREQ = 5_000               # Overwrite latest model every N steps
HIDDEN_LAYERS = [256, 256]             # Policy and value network architecture
DEVICE = "cpu"                         # Device to train on ("cpu", "cuda", or "auto")
FINAL_MODEL_NAME = ""                  # Optional name (no extension). Example: "drone_ppo_run1"

#  Multi-instance Config
NUM_ENVS = 16                           # Number of parallel sims
BASE_PORT = 23000                      # First ZMQ RPC port
PORT_STRIDE = 2                        # Port increment per instance
HOST = "localhost"

#  Optional: Launch CoppeliaSim instances from Python
#  Detects platform and sets paths accordingly
if platform.system() == "Windows":
    LAUNCH_CONFIG = dict(
        enable=False,
        sim_exe_path="C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/coppeliaSim.exe",
        scene_path="C:/Users/jack/Documents/Github/CPSC4420-DroneNAV/CoppeliaSim Drone Follower.ttt",
        headless=False,
        launch_delay=2.0,
    )
else:
    # Linux / Palmetto cluster (inside Apptainer container)
    _project_dir = os.path.dirname(os.path.abspath(__file__))
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
    """
    Custom callback that logs additional metrics during training.

    Tracks collision count, visited grid cells, and minimum LiDAR
    distance per step for TensorBoard monitoring.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        """
        Called at every training step. Extracts info from
        completed episodes and records custom metrics.

        Returns:
            bool: Always True (continue training).
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


class EvalMetricsCallback(BaseCallback):
    """
    Periodically runs deterministic evaluation episodes and logs
    clean metrics to TensorBoard for easy comparison between runs.
    """

    def __init__(self, eval_env, eval_freq, n_eval_episodes=1, seed=0, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.seed = int(seed)
        self._last_eval_timestep = 0

    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self._last_eval_timestep < self.eval_freq:
            return

        self._last_eval_timestep = self.num_timesteps

        rewards = []
        lengths = []
        min_lidars = []
        visited_cells = []
        collisions = 0
        total_steps = 0

        for i in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset(seed=self.seed + i)
            done = False
            ep_reward = 0.0
            ep_len = 0
            ep_min_lidar = float("inf")
            ep_visited = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                ep_reward += reward
                ep_len += 1
                total_steps += 1

                if "min_lidar" in info:
                    ep_min_lidar = min(ep_min_lidar, float(info["min_lidar"]))
                if info.get("collision"):
                    collisions += 1
                if "visited_cells" in info:
                    ep_visited = info["visited_cells"]

                done = terminated or truncated

            rewards.append(ep_reward)
            lengths.append(ep_len)
            min_lidars.append(ep_min_lidar if ep_min_lidar != float("inf") else 0.0)
            visited_cells.append(ep_visited)

        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        mean_len = float(np.mean(lengths)) if lengths else 0.0
        mean_min_lidar = float(np.mean(min_lidars)) if min_lidars else 0.0
        mean_visited = float(np.mean(visited_cells)) if visited_cells else 0.0
        collisions_per_1k = (collisions / total_steps * 1000.0) if total_steps > 0 else 0.0

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_len", mean_len)
        self.logger.record("eval/mean_visited_cells", mean_visited)
        self.logger.record("eval/min_lidar", mean_min_lidar)
        self.logger.record("eval/collisions_per_1k_steps", collisions_per_1k)

    def _on_training_end(self) -> None:
        try:
            self.eval_env.close()
        except Exception:
            pass


class PeriodicSaveCallback(BaseCallback):
    """
    Periodically saves/overwrites a "latest" model file and
    the VecNormalize running statistics alongside it.
    """

    def __init__(self, save_freq, save_path, stats_path=None, verbose=0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        self.stats_path = stats_path

    def _on_step(self) -> bool:
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
    """Returns a zero-arg factory that creates a DroneAvoidanceEnv on *port*."""
    def _init():
        env = DroneAvoidanceEnv(**ENV_CONFIG, host=HOST, port=port)
        return env
    return _init


def build_vec_env(normalize=True):
    """
    Builds a vectorized environment, optionally wrapped in VecNormalize.

    NUM_ENVS >= 2  →  SubprocVecEnv  (one process per sim)
    NUM_ENVS == 1  →  DummyVecEnv    (single process, easier to debug)

    Args:
        normalize: If True (default), wrap with VecNormalize for
            obs + reward normalization.  Pass False when you plan
            to load saved VecNormalize stats via VecNormalize.load().

    Returns:
        VecNormalize or VecMonitor depending on *normalize*.
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
    if not LAUNCH_CONFIG["enable"]:
        return

    sim_exe = LAUNCH_CONFIG["sim_exe_path"]
    scene = LAUNCH_CONFIG["scene_path"]
    if not sim_exe or not scene:
        raise ValueError("LAUNCH_CONFIG requires sim_exe_path and scene_path when enabled.")

    total_instances = NUM_ENVS
    if EVAL_CONFIG["enabled"] and EVAL_CONFIG["use_separate_env"]:
        total_instances += 1

    for i in range(total_instances):
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
    """
    Runs PPO training.

    Creates the environment, builds the PPO model with the
    configured hyperparameters, and trains for TOTAL_TIMESTEPS.
    Saves checkpoints every CHECKPOINT_FREQ steps and a final
    model when training completes or is interrupted with Ctrl+C.

    If resume_path is provided, loads a previously saved model
    and continues training from where it left off.

    Args:
        resume_path (str, optional): Path to a saved model to
            resume training from (without .zip extension).
    """

    os.makedirs("models", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
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

    eval_env = None
    if EVAL_CONFIG["enabled"]:
        if NUM_ENVS > 1 and EVAL_CONFIG["use_separate_env"]:
            eval_port = BASE_PORT + NUM_ENVS * PORT_STRIDE
            eval_env = DroneAvoidanceEnv(**ENV_CONFIG, host=HOST, port=eval_port)
            eval_env = Monitor(eval_env)
        else:
            print("Eval disabled: set EVAL_CONFIG['use_separate_env']=True and NUM_ENVS>1.")

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
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path="./models/checkpoints/",
        name_prefix="drone_ppo",
    )
    metrics_cb = TrainingMetricsCallback()
    eval_cb = None
    if EVAL_CONFIG["enabled"] and eval_env is not None:
        eval_cb = EvalMetricsCallback(
            eval_env=eval_env,
            eval_freq=EVAL_CONFIG["eval_freq"],
            n_eval_episodes=EVAL_CONFIG["n_eval_episodes"],
            seed=EVAL_CONFIG["seed"],
        )
    latest_path = f"models/{run_name}"
    latest_cb = PeriodicSaveCallback(
        save_freq=LATEST_SAVE_FREQ,
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
        callbacks = [checkpoint_cb, metrics_cb, latest_cb]
        if eval_cb is not None:
            callbacks.append(eval_cb)

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
            _safe_close_env(eval_env)
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
    """
    Runs a trained model in a single episode with deterministic actions.

    Builds a VecNormalize-wrapped env in eval mode, loads the saved
    normalization stats alongside the model weights, and prints
    live telemetry every 50 steps.

    Args:
        model_path (str): Path to the saved model file
            (without .zip extension).
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
