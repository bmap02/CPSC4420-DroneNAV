# CoppeliaSim Drone Obstacle Avoidance

Reinforcement learning system that trains a quadcopter in CoppeliaSim to autonomously explore a map and avoid static and dynamic obstacles using PPO and angular-binned LiDAR. The drone learns purely from experience — no waypoints, no scripted paths, no hand-coded avoidance rules.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [CoppeliaSim Setup](#coppeliasim-setup)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Palmetto Cluster Setup](#palmetto-cluster-setup-clemson-hpc)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- [CoppeliaSim](https://www.coppeliarobotics.com/) EDU or Player (V4.10.0+)
- Python 3.10+
- GPU not required — training bottleneck is CoppeliaSim physics (CPU-bound), not the neural network

## Installation

### Local (Windows / Mac)

```bash
pip install -r requirements.txt
```

### Palmetto (Clemson HPC)

See [Palmetto Cluster Setup](#palmetto-cluster-setup-clemson-hpc).

## CoppeliaSim Setup

### Installing CoppeliaSim (quick start)

1. **Download** the EDU (free for academic use) or Player edition from the [official downloads page](https://www.coppeliarobotics.com/downloads). This project targets **V4.10.0** — newer versions should work but haven't been tested.
2. **Install** using the installer for your OS (Windows `.exe`, Mac `.dmg`, or Linux `.tar.xz`).
3. **Launch** CoppeliaSim and confirm it opens to an empty scene without errors.

### Opening the project scene

1. In CoppeliaSim, go to **File → Open Scene…**
2. Navigate to the project folder and select **`CoppeliaSim Drone Follower.ttt`**
3. Leave the simulation **stopped** (do not press the ▶ Start button) — the Python scripts will start and stop the sim automatically. You can see in the object hierarchy on the left that the scene contains `/Quadcopter`, `/target`, several trees, buildings, and walking humanoids.

> **Before running any command** that talks to CoppeliaSim (`python train.py`, `python evaluate.py`, `python generate_spawn_map.py`, or `python train.py --test`), make sure CoppeliaSim is **open** with the scene file loaded. The Python side connects to the running CoppeliaSim process via ZMQ on port 23000.

### Required scene contents

The `CoppeliaSim Drone Follower.ttt` scene must contain:

- `/Quadcopter` — drone model with a built-in flight script that chases a target dummy
- `/target` — dummy the flight script follows (must be *non-detectable* so LiDAR ignores it)
- `/Quadcopter/base/lidarFront/body/sensor` — front LiDAR vision sensor
- `/Quadcopter/base/lidarBack/body/sensor` — back LiDAR vision sensor
- `/Quadcopter/base/lidarLeft/body/sensor` — left LiDAR vision sensor
- `/Quadcopter/base/lidarRight/body/sensor` — right LiDAR vision sensor
- Static obstacles (trees, buildings, etc.) and dynamic obstacles (animated people on paths)

**Important:** each LiDAR sensor's FOV must be **90 degrees**. The Lua script under each `lidarX` object sets this in `sysCall_init` (`scanningAngle` variable). Without 90° FOV, there are diagonal blind spots between sensors.

## Quick Start

### 1. Generate the spawn map (one-time per scene)

```bash
python generate_spawn_map.py
```

Sweeps the drone (frozen, flight script disabled) across the map at 0.5 m intervals. Saves obstacle-free positions to `spawn_map.npy` and a human-readable grid to `spawn_map_preview.txt`. Must be regenerated if the scene layout changes.

### 2. Train

```bash
python train.py --final-name my_run
```

Launches `NUM_ENVS` headless CoppeliaSim instances and trains PPO. The model (`models/my_run.zip`) and VecNormalize stats (`stats/my_run_vecnormalize.pkl`) are saved every `SAVE_FREQ` steps and on completion/interrupt.

Monitor training with TensorBoard:

```bash
tensorboard --logdir ./logs/
```

### 3. Evaluate

```bash
python evaluate.py --model models/my_run --episodes 20
```

Runs deterministic episodes (no exploration noise) with frozen VecNormalize stats. Reports per-episode reward, length, collision/OOB/timeout rates, and cells visited. Use `--compare` to evaluate multiple models side-by-side:

```bash
python evaluate.py --compare models/my_run_v1 models/my_run_v2 --episodes 30
```

### 4. Compare training runs

```bash
python compare_runs.py                       # all palmetto runs
python compare_runs.py --pattern v14 v15     # specific versions
python compare_runs.py --mode drift          # spot late-training collapse
```

### 5. Test (quick single-episode demo)

```bash
python train.py --test --model models/my_run
```

### 6. Resume training

```bash
python train.py --resume --model models/my_run
```

**Important:** do *not* resume after changing reward weights or observation shape in `ENV_CONFIG`. The value function is calibrated to the reward scale it was trained on — resuming into a different configuration will collapse the policy. Retrain from scratch instead.

## How It Works

### Observation (22D, normalized by VecNormalize)

| Index | Field | Units |
|---|---|---|
| `[0:16]` | 16 LiDAR angular bins (4 bins × 4 sensors: F, B, L, R) | m |
| `[16:19]` | Drone world position (x, y, z) | m |
| `[19:22]` | Drone linear velocity (vx, vy, vz) | m/s |

Each sensor's 16×16 depth buffer (90° FOV) is split into 4 column bins (~22.5° per bin). The minimum valid depth per bin is reported (readings ≤ 0.01 m are filtered, max range is 5.0 m). `VecNormalize` wraps the environment to normalize observations and rewards to approximately zero-mean, unit-variance.

### Action (3D, continuous)

`(dx, dy, dz)` in `[-1, 1]`, clipped and multiplied by `speed_scale`, then added to the target dummy's world position. The drone's built-in PID flight script chases the dummy. The `z` component is clamped to `[min_altitude, max_altitude]`.

### Reward

All reward weights are configurable via `ENV_CONFIG` in `train.py`. The environment exposes them as **required keyword-only arguments** — forgetting one raises `TypeError`.

| Term | Type | Purpose |
|---|---|---|
| `survival_reward` | per-step `+` | Reward for every step the episode continues |
| `exploration_reward` | one-shot `+` | Bonus the first time the drone enters each 0.5 m grid cell (2D, x/y only) |
| `altitude_bonus` | per-step `+` | Reward when altitude is within the soft band around ideal |
| `proximity_penalty` | per-step `−` | Quadratic penalty when `min_lidar < proximity_threshold` |
| `boundary_penalty` | per-step `−` | Linear ramp within `boundary_warning_distance` of x/y map edges |
| `stagnation_penalty` | per-step `−` | Accrues after camping in one grid cell too long |
| `altitude_penalty` | per-step `−` | Linear then quadratic outside the altitude soft band |
| `action_smoothness` | per-step `−` | `scale × ‖a_t − a_{t-1}‖²` — suppresses twitchy output |
| `movement_reward` | per-step `+` | Reward when horizontal speed > 0.1 m/s (currently `0.0`) |
| `hovering_penalty` | per-step `−` | Penalty when horizontal speed < 0.1 m/s (currently `0.0`) |
| `collision_penalty` | terminal `−` | Episode ends when `min_lidar < collision_distance` |
| `out_of_bounds_penalty` | terminal `−` | Episode ends when drone leaves the flight volume |

### Episode Termination

- **Terminated** (no value bootstrap): collision or out-of-bounds
- **Truncated** (with value bootstrap): `step_count >= max_steps`

## Project Structure

```
CPSC4420-DroneNAV/
├── train.py                    # Training / testing entry point
├── drone_environment.py        # Gymnasium environment wrapping CoppeliaSim
├── lidar.py                    # LiDAR depth reader with angular binning
├── navigation.py               # Position / velocity helpers, target-dummy mover
├── evaluate.py                 # Post-training deterministic evaluation
├── compare_runs.py             # TensorBoard run comparison tables
├── generate_spawn_map.py       # One-time safe-spawn position sweep
├── coppeliasim.def             # Apptainer container definition for Palmetto
├── requirements.txt            # Python dependencies
├── CoppeliaSim Drone Follower.ttt  # Training scene
├── spawn_map.npy               # Pre-computed safe (x, y) spawn positions
├── spawn_map_preview.txt       # Human-readable spawn map grid
├── models/                     # Saved model weights (.zip)
├── stats/                      # VecNormalize running statistics (.pkl)
├── logs/                       # TensorBoard event files
└── archive/                    # Pre-v6 artifacts (gitignored)
```

## Configuration

All configuration lives at the top of `train.py`. Values shown below are the current defaults.

### Environment Parameters (`ENV_CONFIG`)

| Parameter | Value | Description |
|---|---|---|
| `max_steps` | `2000` | Max steps per episode before truncation |
| `speed_scale` | `0.1` | Multiplier from raw action to target-dummy displacement |
| `collision_distance` | `0.1` | LiDAR distance (m) that triggers collision termination |
| `proximity_threshold` | `0.4` | Distance (m) below which proximity penalty applies |
| `boundary_min` / `boundary_max` | `-9.0` / `9.0` | Flight volume x/y bounds (m) |
| `min_altitude` / `max_altitude` | `0.5` / `2.5` | Allowed altitude range (m) |
| `flight_height` | `1.5` | Starting altitude on reset (m) |
| `ideal_altitude` | `1.5` | Ideal cruising altitude (m) |
| `exploration_grid_size` | `0.5` | Grid cell size for exploration tracking (m) |
| `lidar_bins` | `4` | Angular bins per sensor (4 × 4 sensors = 16 features) |
| `randomize_start_pose` | `True` | Random spawn from spawn map each reset |
| `spawn_map_path` | `spawn_map.npy` | Pre-computed safe spawn positions |
| `lidar_resolution` | `16` | Vision sensor resolution (square, pixels) |

### Reward Weights (`ENV_CONFIG`)

| Parameter | Value | Description |
|---|---|---|
| `survival_reward` | `0.3` | Per-step bonus for staying alive |
| `exploration_reward` | `20.0` | One-shot bonus for visiting a new grid cell |
| `proximity_penalty_scale` | `25.0` | Quadratic proximity penalty multiplier |
| `collision_penalty` | `60.0` | Terminal penalty for collision |
| `out_of_bounds_penalty` | `50.0` | Terminal penalty for leaving flight volume |
| `altitude_bonus` | `0.5` | Bonus inside the altitude soft band |
| `altitude_soft_band` | `0.5` | Half-width of the free-altitude zone (m) |
| `altitude_linear_scale` | `1.0` | Linear altitude penalty slope |
| `altitude_quadratic_scale` | `2.0` | Quadratic altitude penalty slope |
| `action_smoothness_scale` | `0.02` | Penalty on `‖a_t − a_{t-1}‖²` |
| `boundary_warning_distance` | `0.7` | Distance from edge where boundary penalty starts (m) |
| `boundary_penalty_scale` | `2.0` | Boundary proximity penalty multiplier |

### PPO Hyperparameters (`PPO_CONFIG`)

| Parameter | Value | Description |
|---|---|---|
| `learning_rate` | `3e-4` | Adam learning rate |
| `n_steps` | `256` | Steps per env per rollout (× `NUM_ENVS` = total samples) |
| `batch_size` | `64` | Minibatch size per gradient step |
| `n_epochs` | `5` | SGD passes over each rollout buffer |
| `gamma` | `0.995` | Discount factor (~200-step effective horizon) |
| `gae_lambda` | `0.95` | GAE lambda for advantage estimation |
| `clip_range` | `0.2` | PPO clipping range |
| `ent_coef` | `0.005` | Entropy bonus (exploration vs commitment) |
| `vf_coef` | `0.5` | Value function loss weight |
| `max_grad_norm` | `0.5` | Gradient clipping threshold |
| `target_kl` | `0.04` | Stop update early if approx KL exceeds this |

### Training Constants

| Constant | Value | Description |
|---|---|---|
| `TOTAL_TIMESTEPS` | `500_000` | Total training timesteps |
| `SAVE_FREQ` | `5_000` | Save model + stats every N steps |
| `HIDDEN_LAYERS` | `[256, 256]` | Policy / value network architecture (ReLU MLP) |
| `NUM_ENVS` | `16` | Parallel CoppeliaSim instances (set to 1 for local testing) |
| `DEVICE` | `"cpu"` | Training device (`"cpu"`, `"cuda"`, or `"auto"`) |

## Palmetto Cluster Setup (Clemson HPC)

Palmetto is Clemson's HPC cluster. Training runs on compute nodes inside an [Apptainer](https://apptainer.org/) container that bundles CoppeliaSim + all Python dependencies into a single portable file. This section walks through the complete setup from scratch.

### What you need before starting

- A Clemson account with Palmetto access
- The project repo on GitHub (your fork or the main repo)
- A local machine for viewing TensorBoard logs and testing models

### Understanding the tools

| Tool | What it is | Why we need it |
|---|---|---|
| **Open OnDemand** | Web portal for Palmetto (https://docs.rcd.clemson.edu/openod/) | Launch interactive sessions without SSH |
| **Palmetto Desktop** | Full Linux desktop on a compute node (via OnDemand) | Run terminal commands, build containers, train |
| **Code Server** | VSCode in the browser (via OnDemand) | Edit files on Palmetto — but can't run `apptainer` commands |
| **Apptainer** | Container runtime (like Docker but for HPC) | Packages CoppeliaSim + Python into a portable `.sif` file |
| `coppeliasim.def` | Container recipe (in the repo) | Tells Apptainer what to download and install |
| `coppeliasim.sif` | Built container (~1–2 GB) | The actual portable environment — run training inside this |

### Step 1: Launch a Palmetto Desktop

1. Go to [Open OnDemand](https://docs.rcd.clemson.edu/openod/)
2. Click **Interactive Apps → Palmetto Desktop**
3. Set these parameters:

   | Parameter | Value | Why |
   |---|---|---|
   | Partition | `work1` | General compute |
   | CPU cores | **20** | ~1–2 cores per CoppeliaSim instance |
   | Memory | **64–80 GB** | ~2 GB per instance × 16 instances + overhead |
   | GPUs | **0** | Not needed — bottleneck is CoppeliaSim physics |
   | Hours | **3–4** (setup) or **12+** (training) | Container build is one-time; training duration depends on `TOTAL_TIMESTEPS` and `NUM_ENVS` |

4. Click **Launch** and wait for the session to start
5. Click **Launch Palmetto Desktop** to open the Linux desktop
6. Right-click the desktop → **Open Terminal**

> **Important:** Do NOT use Code Server for terminal commands. It runs inside its own container where `apptainer` is not available. Use Code Server only for editing files.

### Step 2: Clone the repository

```bash
mkdir -p ~/rl_testing && cd ~/rl_testing
git clone https://github.com/YOUR_USERNAME/CPSC4420-DroneNAV.git
cd CPSC4420-DroneNAV
```

### Step 3: Build the Apptainer container (one-time)

```bash
apptainer build --fakeroot coppeliasim.sif coppeliasim.def
```

This downloads Ubuntu 24.04, CoppeliaSim V4.10.0, PyTorch, stable-baselines3, and all other dependencies, then packages them into `coppeliasim.sif`.

- `--fakeroot` lets you build without root access (available on Palmetto)
- You only need to rebuild if you change `coppeliasim.def` or need to update packages
- The `.sif` file is ~1–2 GB — do NOT commit it to git

**Verify it worked:**

```bash
apptainer exec coppeliasim.sif xvfb-run /opt/coppeliasim/coppeliaSim -h 2>&1 | head -20
```

You should see:
```
[CoppeliaSim:loadinfo]   simulator launched.
[CoppeliaSim:loadinfo]   plugin 'RemoteApi': load succeeded.
```

### Step 4: Generate the spawn map (one-time per scene)

```bash
apptainer exec -B $PWD:/workspace coppeliasim.sif bash -c "cd /workspace && python3 generate_spawn_map.py"
```

- `-B $PWD:/workspace` makes your project folder visible inside the container at `/workspace`
- Creates `spawn_map.npy` and `spawn_map_preview.txt`
- Only needs to be regenerated if the scene layout changes

### Step 5: Configure training

Before your first training run, update `NUM_ENVS` in `train.py` to match your core count (16 is recommended for 20 cores). Set `TOTAL_TIMESTEPS` to your desired training budget.

### Step 6: Run training

```bash
apptainer exec -B $PWD:/workspace coppeliasim.sif bash -c "cd /workspace && python3 train.py --final-name drone_ppo_palmetto_v1"
```

**What happens:**
1. `train.py` auto-launches 16 headless CoppeliaSim instances (each wrapped in `xvfb-run`)
2. PPO collects experience from all 16 environments in parallel
3. Model + VecNormalize stats are saved every 5,000 steps
4. TensorBoard logs are written to `logs/{run_name}/`

**To interrupt gracefully:** press `Ctrl+C` once. The model will be saved before shutdown.

### Step 7: Push results to Git

After training completes, commit and push the outputs so you can view them on your local machine:

```bash
cd ~/rl_testing/CPSC4420-DroneNAV

# Stage the training outputs
git add models/drone_ppo_palmetto_v1.zip
git add stats/drone_ppo_palmetto_v1_vecnormalize.pkl
git add -f logs/drone_ppo_palmetto_v1_1/   # -f because logs/ is gitignored

# Commit and push
git commit -m "Training run: drone_ppo_palmetto_v1 (500K steps)"
git push
```

> **Note:** `logs/` and `stats/` are in `.gitignore` to prevent accidental commits during development. Use `git add -f` to force-add specific run outputs you want to share.

### Step 8: View results on your local machine

On your local machine, pull the results:

```bash
git pull
```

**View TensorBoard logs:**
```bash
tensorboard --logdir ./logs/
```
Open http://localhost:6006 in your browser.

**Compare training runs:**
```bash
python compare_runs.py
```

**Evaluate a model** (requires CoppeliaSim running locally):
```bash
python evaluate.py --model models/drone_ppo_palmetto_v1 --episodes 20
```

### Resource reference

| What | How much | Notes |
|---|---|---|
| RAM per CoppeliaSim instance | ~2 GB | 16 instances ≈ 30 GB |
| CPU per instance | ~1–2 cores | Scale `NUM_ENVS` to match available cores |
| GPU | Not needed | PPO MLP is tiny; CoppeliaSim is CPU-only |
| Container size | ~1–2 GB | Do not commit to git |

## Known Limitations

- **No temporal observation.** The policy sees only the current timestep — no memory. It cannot predict pedestrian velocity, only react to instantaneous distance. Fix: `VecFrameStack(k=4)` or `RecurrentPPO`.
- **LiDAR angular resolution.** 4 bins per sensor (~22.5° each) partially addresses bearing info but finer resolution could improve obstacle localization.

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Training hangs on startup | CoppeliaSim already running on same port | Close existing instances or change `BASE_PORT` |
| `The paging file is too small` | Not enough RAM for `NUM_ENVS` instances | Reduce `NUM_ENVS` (each uses ~2 GB) |
| `spawn_map.npy` not found | Spawn map not generated | Run `python generate_spawn_map.py` with CoppeliaSim open |
| `TypeError: missing keyword argument` | `ENV_CONFIG` missing a required param | Add the missing key to `ENV_CONFIG` in `train.py` |
| `approx_kl` rising throughout training | LR too high or `target_kl` too tight | Lower `learning_rate`, raise `target_kl`, or reduce `n_epochs` |
| Model acts randomly when tested | VecNormalize stats not loaded | Ensure `stats/{name}_vecnormalize.pkl` exists alongside the model |
| `apptainer: command not found` on Palmetto | Using Code Server instead of Palmetto Desktop | Switch to Palmetto Desktop for terminal commands |
| Import hangs on Palmetto | Leftover CoppeliaSim processes from previous run | Cancel job and request a new node |
