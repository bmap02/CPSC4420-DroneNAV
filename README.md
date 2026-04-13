# CoppeliaSim Drone Obstacle Avoidance

Reinforcement learning for a quadcopter in CoppeliaSim that learns to explore a map and avoid static and dynamic obstacles using PPO and 4-direction LiDAR. The drone learns from experience — no waypoints, no scripted paths, no hand-coded avoidance rules. The policy's action is the sole source of control.

## Prerequisites

- [CoppeliaSim](https://www.coppeliarobotics.com/) (EDU or Player), 4.5+
- Python 3.10+
- NVIDIA GPU optional (CPU training works fine for the current network size)

## Installation

```bash
pip install coppeliasim-zmqremoteapi-client numpy gymnasium stable-baselines3 tensorboard rich tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## CoppeliaSim Scene Setup

The scene (`CoppeliaSim Drone Follower.ttt`) must contain:

- `/Quadcopter` — drone model with a built-in flight script that chases a target dummy
- `/target` — dummy the flight script follows (must be *non-detectable* so LiDAR ignores it)
- `/Quadcopter/base/lidarFront/body/sensor` — front LiDAR vision sensor
- `/Quadcopter/base/lidarBack/body/sensor` — back LiDAR vision sensor
- `/Quadcopter/base/lidarLeft/body/sensor` — left LiDAR vision sensor
- `/Quadcopter/base/lidarRight/body/sensor` — right LiDAR vision sensor
- Static obstacles (trees, buildings, etc.) and dynamic obstacles (animated people on paths)

## Usage

### Training

```bash
python train.py
```

`train.py` auto-launches `NUM_ENVS` headless CoppeliaSim instances (16 by default), each bound to its own ZMQ port, and trains PPO against them in parallel. Checkpoints go to `./models/checkpoints/`, and the final model to `./models/drone_ppo_final.zip`. Monitor with TensorBoard:

```bash
tensorboard --logdir ./logs/
```

### Testing a trained model

```bash
python train.py --test
python train.py --test --model models/checkpoints/drone_ppo_50000_steps
```

### Resuming training

```bash
python train.py --resume --model models/drone_ppo_final
python train.py --resume --model models/checkpoints/drone_ppo_10000_steps
```

**Important:** do *not* resume an old checkpoint after changing reward weights in `ENV_CONFIG`. PPO's value function is calibrated to the reward scale it was trained on; resuming into a different reward produces wildly incorrect advantages and the policy collapses. Retrain from scratch whenever you touch the reward.

### Running on Palmetto (Clemson HPC)

```bash
cd ~/rl_testing/CPSC4420-DroneNAV
apptainer exec -B $PWD:/workspace coppeliasim.sif bash -c "cd /workspace && python3 train.py --final-name drone_ppo_palmetto_v5"
```

## How it works

### Observation (10D, flat Box, unnormalized)

| Index | Field | Units |
|---|---|---|
| `[0:4]` | 4 LiDAR min-distances (Front, Back, Left, Right) | m |
| `[4:7]` | Drone world position (x, y, z) | m |
| `[7:10]` | Drone linear velocity (vx, vy, vz) | m/s |

Each "LiDAR" is a 32×32 depth image from a CoppeliaSim vision sensor, reduced to its minimum valid depth (readings below `0.01 m` are filtered out, max range is `5.0 m`).

### Action (3D, continuous)

`(dx, dy, dz)` in `[-1, 1]`, clipped and multiplied by `speed_scale`, then added to the target dummy's world position. The drone's built-in flight script chases the dummy. The `z` component is clamped to `[min_altitude, max_altitude]` so the policy can't cheat by climbing out of obstacle range.

### Reward

Every reward weight is configurable from `train.py`'s `ENV_CONFIG` — `drone_environment.py` exposes all of them as **required keyword-only arguments**, so forgetting to set one raises a `TypeError` instead of silently picking a default. This guarantees `train.py` is the single source of truth.

| Term | Type | Purpose |
|---|---|---|
| `survival_reward` | per-step `+` | Reward for every step the episode continues |
| `exploration_reward` | one-shot `+` | Bonus the first time the drone enters each 0.5 m grid cell |
| `movement_reward` | per-step `+` | Reward when horizontal speed `> 0.1 m/s` (currently `0`) |
| `altitude_bonus` | per-step `+` | Reward when within the altitude soft band |
| `proximity_penalty` | per-step `−` | Quadratic or linear in `(proximity_threshold − min_lidar)` |
| `stagnation_penalty` | per-step `−` | Accrues after camping in one cell for too long |
| `altitude_penalty` | per-step `−` | Linear then quadratic outside the soft band |
| `hovering_penalty` | per-step `−` | Penalty when horizontal speed `< 0.1 m/s` (currently `0`) |
| `action_smoothness` | per-step `−` | `scale × ‖a_t − a_{t-1}‖²` to suppress twitchy output |
| `collision_penalty` | terminal `−` | Triggered when `min_lidar < collision_distance` |
| `out_of_bounds_penalty` | terminal `−` | Triggered when the drone leaves the flight volume |

The reward has been deliberately rebalanced (see *Reward design notes* below) to remove the "fly a big perimeter circle forever" attractor that dominated earlier runs.

### Episode termination

- **Terminated** (no value bootstrap): collision (`min_lidar < collision_distance`) or out-of-bounds (position outside `[boundary_min, boundary_max]` on x/y or `[min_altitude, max_altitude]` on z).
- **Truncated** (value bootstrap): `step_count >= max_steps`.

### Training loop

`train.py` uses stable-baselines3 PPO with a `[256, 256]` ReLU MLP and 16 `SubprocVecEnv` workers, each talking to its own CoppeliaSim instance over ZMQ. Default total budget is 500 k timesteps. Checkpoints are written every 10 k steps, and a "latest" model every 5 k steps so in-progress training is always recoverable.

## Reward design notes

The reward went through a ground-up rebalance after the original weights produced a "big perimeter circle" policy. This section documents what was changed and why, because the numbers in `ENV_CONFIG` don't explain themselves.

### What the old reward optimized

The previous reward had `movement_reward = +4.0/step` awarded for *any* horizontal speed above `0.1 m/s`, regardless of direction or displacement. On a 2000-step episode, this was worth up to `+8000` of episode reward — an order of magnitude larger than the collision penalty (`−60`) and more than triple the maximum possible exploration reward. The optimal policy had two phases:

1. **Lap 1** (~570 steps): sweep the perimeter once, collecting ~400 new cells at `+20` each. Average reward per step `≈ +14.8`.
2. **Laps 2+** (remaining ~1430 steps): keep looping the same perimeter. Exploration reward is zero (cells are one-shot), but `movement_reward + survival + altitude_bonus = +4.8/step` flows forever. No penalty because the interior (where the obstacles are) is avoided.

Total episode reward: `~+17,560`. The policy was not "exploring" — it was running one efficient sweep and then farming the movement bonus.

### What was changed

| Term | Old | New | Why |
|---|---|---|---|
| `movement_reward` | `4.0` | `0.0` | Laps 2+ now pay nothing. The big circle is no longer profitable after the first lap. |
| `proximity_threshold` | `1.0 m` | `0.25 m` | Narrow corridors with walls at ≥ 0.25 m are penalty-free. The drone can thread gaps instead of avoiding the interior. |
| `proximity_penalty_scale` | `3.5` (linear) | `25.0` (quadratic) | With the threshold tightened, the penalty scale was raised and the shape changed to `((threshold − d)/threshold)²` so near-collision is sharply punished. |
| `hovering_penalty` | `1.0` | `0.0` | Removed alongside the movement bonus. Keeping it created a one-sided cliff at `speed = 0.1` with nothing on the other side — just noise for the value function to learn. |
| `altitude_soft_band` | `0.3 m` | `0.5 m` | Widened so the drone has room to change altitude without penalty. |
| `altitude_linear_scale` | `1.5` | `0.3` | Softened. The old value made altitude the dominant gradient in the reward. |
| `altitude_quadratic_scale` | `3.0` | `0.5` | Same. |
| `action_smoothness_scale` | — | `0.02` | New term: `0.02 × ‖a_t − a_{t-1}‖²`. Suppresses the "twitch in place to satisfy the speed threshold" reward-hack that could replace the big circle. |

### Near-collision penalty profile

With `proximity_threshold = 0.25 m`, `scale = 25`, quadratic, the per-step reward near obstacles looks like this (with the `+0.8/step` base from survival + altitude):

| Lidar distance | Net reward/step | Interpretation |
|---|---|---|
| `≥ 0.25 m` | `+0.8` | Free zone — any corridor `≥ 0.5 m` wide is penalty-free |
| `0.22 m` | `+0.44` | Mild warning |
| `0.20 m` | `−0.20` | Mild warning |
| `0.15 m` | `−3.20` | Danger zone |
| `0.12 m` | `−5.96` | Imminent collision |
| `< 0.10 m` | `−60` + terminal | Collision, episode ends |

### What this change does and doesn't fix

**Does fix:** the reward landscape no longer pays for circling, oscillation, or ramming walls. Interior sweep is worth ~50 % more than the big perimeter circle. Near-collision is unambiguously negative for the first time.

**Does not fix:** the *policy-space* attractor. With a fixed start pose, all rollouts begin at the same point, and PPO's initial exploration strategy (the big perimeter circle) is a strong local optimum that's hard to escape even when it's no longer globally optimal. Breaking that attractor requires *randomizing the spawn pose within the map* so PPO sees rollouts that start in the interior — planned as a future change.

**Observability gap (unchanged):** dynamic pedestrians move, but the observation has no memory. The policy sees only the instantaneous distance to each obstacle, so it cannot form a velocity estimate and cannot predict where a pedestrian will be next step. This limits "predictive" avoidance to reflexive distance-based avoidance until the observation adds temporal context (frame stacking or a recurrent policy).

## Structure

| File | Description |
|---|---|
| `train.py` | PPO training / testing entry point. All tunable config (`ENV_CONFIG`, `PPO_CONFIG`, `EVAL_CONFIG`, launcher paths) lives at the top of the file. |
| `drone_environment.py` | Gymnasium environment wrapping CoppeliaSim. All reward and env kwargs are required keyword-only — supplied by `train.py`. |
| `lidar.py` | Reads and formats depth data from the 4 vision-based LiDAR sensors. |
| `navigation.py` | Position / velocity getters, distance helper, and target-dummy mover. |
| `CoppeliaSim Drone Follower.ttt` | The training scene. |

## Configuration

### Environment parameters (`ENV_CONFIG` in `train.py`)

| Parameter | Value | Description |
|---|---|---|
| `max_steps` | `2000` | Max steps per episode before truncation |
| `speed_scale` | `0.1` | Multiplier applied to policy action before it's sent to the target dummy |
| `collision_distance` | `0.1` | LiDAR distance (m) that triggers collision termination |
| `proximity_threshold` | `0.25` | LiDAR distance (m) below which the proximity penalty applies |
| `boundary_min` / `boundary_max` | `-9.0` / `9.0` | Flight volume x/y bounds (m) |
| `min_altitude` / `max_altitude` | `0.3` / `3.0` | Allowed altitude range (m) |
| `flight_height` | `1.5` | Starting altitude on reset (m) |
| `ideal_altitude` | `1.5` | Ideal cruising altitude (m) |
| `exploration_grid_size` | `0.5` | Grid cell size for exploration tracking (m) |
| `disable_visualization` | `True` | Disable sim display on reset (required for headless training) |
| `lidar_resolution` | `32` | Vision sensor resolution (square) |

### Reward weights (`ENV_CONFIG` in `train.py`)

| Parameter | Value | Description |
|---|---|---|
| `survival_reward` | `0.3` | Per-step bonus for staying alive |
| `exploration_reward` | `20.0` | One-shot bonus for visiting a new grid cell |
| `movement_reward` | `0.0` | Per-step bonus when horizontal speed > 0.1 m/s (disabled) |
| `stagnation_start` | `20` | Steps in one cell before stagnation penalty kicks in |
| `stagnation_rate` | `0.3` | Per-step stagnation accrual after it starts |
| `stagnation_cap` | `8.0` | Max stagnation penalty per step |
| `proximity_penalty_scale` | `25.0` | Multiplier on the proximity penalty |
| `proximity_penalty_quadratic` | `True` | Use quadratic shape `((T−d)/T)² × scale` |
| `collision_penalty` | `60.0` | Terminal penalty for collision |
| `out_of_bounds_penalty` | `50.0` | Terminal penalty for leaving the flight volume |
| `hovering_penalty` | `0.0` | Per-step penalty when speed < 0.1 m/s (disabled) |
| `altitude_bonus` | `0.5` | Bonus when within the altitude soft band |
| `altitude_soft_band` | `0.5` | Half-width of the "free altitude" band (m) |
| `altitude_linear_band` | `1.0` | Beyond this Δ, switch to quadratic penalty (m) |
| `altitude_linear_scale` | `0.3` | Linear penalty slope inside the linear band |
| `altitude_quadratic_scale` | `0.5` | Quadratic penalty slope outside the linear band |
| `action_smoothness_scale` | `0.02` | Penalty on `‖a_t − a_{t-1}‖²` |

### PPO hyperparameters (`PPO_CONFIG` in `train.py`)

| Parameter | Value | Description |
|---|---|---|
| `learning_rate` | `5e-4` | Learning rate (Adam) |
| `n_steps` | `256` | Steps per env per rollout (× 16 envs = 4096-sample rollouts) |
| `batch_size` | `64` | Minibatch size for each gradient step |
| `n_epochs` | `10` | PPO update epochs per rollout |
| `gamma` | `0.98` | Discount factor |
| `gae_lambda` | `0.95` | GAE lambda for advantage estimation |
| `clip_range` | `0.2` | PPO clipping range |
| `ent_coef` | `0.01` | Entropy coefficient |
| `vf_coef` | `0.5` | Value function loss weight |
| `max_grad_norm` | `0.5` | Gradient clipping threshold |

### Training budget (`train.py` constants)

| Constant | Value | Description |
|---|---|---|
| `TOTAL_TIMESTEPS` | `500_000` | Total training timesteps |
| `CHECKPOINT_FREQ` | `10_000` | Save a versioned checkpoint every N steps |
| `LATEST_SAVE_FREQ` | `5_000` | Overwrite the "latest" model every N steps |
| `HIDDEN_LAYERS` | `[256, 256]` | Policy / value network architecture |
| `NUM_ENVS` | `16` | Parallel CoppeliaSim instances |
| `BASE_PORT` / `PORT_STRIDE` | `23000` / `2` | First ZMQ port and increment per instance |
| `DEVICE` | `"cpu"` | Training device (`"cpu"`, `"cuda"`, or `"auto"`) |

## Known limitations

These are tracked in `.claude/plans/dazzling-questing-boole.md` (project-internal critique) and will be addressed in subsequent changes:

- **Fixed spawn pose.** All rollouts start at the same point. Even with the reward rebalance, the big-circle attractor is still a strong local optimum in policy space. Fix: randomize start pose within the map on each `reset()`.
- **No temporal observation.** The policy has no memory, so it cannot predict pedestrian motion — only react to it. Fix: `VecFrameStack(k=4)` or switch to `RecurrentPPO`.
- **No `VecNormalize`.** Observations are unbounded and rewards are large-magnitude. PPO's value function is harder to fit than it needs to be. Fix: wrap the vec env with `VecNormalize(norm_obs=True, norm_reward=True)` and persist the stats alongside the model.
- **Eval is disabled.** `EVAL_CONFIG["enabled"] = False`. There's no objective generalization metric in TensorBoard. Fix: enable the eval callback (and fix the port-collision bug at `train.py:359` while you're in there).
- **Lossy LiDAR.** Each 32×32 depth image is reduced to its minimum, throwing away all bearing information within the sensor's wedge. Fix: angular-binned lidar feature (e.g. 8 bins × 4 sensors = 32 features) with optional small encoder.
