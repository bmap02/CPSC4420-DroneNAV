"""Deterministic evaluation of a trained drone model.

Runs N episodes with ``deterministic=True`` (no exploration noise)
and ``VecNormalize`` in eval mode (running stats are frozen -- no
updates).  Aggregates per-episode metrics so you can tell:

- Is the policy reliable, or is its performance a fluke?
- How does it behave across random spawn positions?
- Is it overfitting to training-time exploration noise?

Key differences from training:

- **Deterministic actions** -- ``model.predict(deterministic=True)``
  removes the Gaussian noise that PPO adds during training.
- **Frozen normalisation** -- ``env.training = False`` prevents the
  running mean/std from drifting during evaluation.
- **Raw rewards** -- ``env.norm_reward = False`` so reported reward
  numbers reflect the true (un-normalised) reward signal.

Usage::

    python evaluate.py                                           # default model
    python evaluate.py --model models/drone_ppo_palmetto_v14     # specific model
    python evaluate.py --episodes 50                             # more episodes
    python evaluate.py --compare models/v14 models/v15           # side-by-side

The model path must NOT include the ``.zip`` extension (SB3 convention).
The ``VecNormalize`` stats are loaded from
``stats/{model_name}_vecnormalize.pkl``.

Requires one CoppeliaSim instance running on ``BASE_PORT`` (23000).
"""

import argparse
import os
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from drone_environment import DroneAvoidanceEnv
from train import ENV_CONFIG, HOST, BASE_PORT


def build_eval_env():
    """Builds a single-env ``DummyVecEnv`` for evaluation.

    Ignores ``train.py``'s ``NUM_ENVS`` -- evaluation always runs
    one episode at a time against one CoppeliaSim instance on
    ``BASE_PORT``.

    Returns:
        A ``VecMonitor``-wrapped ``DummyVecEnv`` containing one
        ``DroneAvoidanceEnv``.
    """
    def _init():
        return DroneAvoidanceEnv(**ENV_CONFIG, host=HOST, port=BASE_PORT)

    vec = DummyVecEnv([_init])
    vec = VecMonitor(vec)
    return vec


def eval_single(model_path, n_episodes=20, seed=42, verbose=True):
    """Runs *n_episodes* deterministic episodes on a saved model.

    Args:
        model_path: Path to the saved model (without ``.zip``).
        n_episodes: Number of evaluation episodes.
        seed: Base RNG seed for reproducible spawn positions.
        verbose: If ``True``, print per-episode results.

    Returns:
        A dict of per-episode arrays and summary statistics
        including reward, length, collision/OOB/timeout rates,
        visited cells, and closest LiDAR reading.
    """
    run_name = os.path.basename(model_path)
    stats_path = f"stats/{run_name}_vecnormalize.pkl"

    if verbose:
        print(f"\n=== Evaluating: {model_path} ===")

    # Build env and load normalization stats (if available)
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, build_eval_env())
        if verbose:
            print(f"  Stats: {stats_path}")
    else:
        print(f"  WARNING: no VecNormalize stats at {stats_path}")
        print("  Observations will be re-normalized during eval which is wrong")
        env = VecNormalize(build_eval_env(), norm_obs=True, norm_reward=False, clip_obs=10.0)

    env.training = False      # don't update running stats
    env.norm_reward = False   # raw rewards for reporting

    model = PPO.load(model_path, env=env)
    if verbose:
        print("  Loaded model")

    # Per-episode metric buffers
    rewards = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=int)
    collisions = np.zeros(n_episodes, dtype=int)
    oob_terminations = np.zeros(n_episodes, dtype=int)
    timeout_terminations = np.zeros(n_episodes, dtype=int)
    min_lidars = np.zeros(n_episodes)
    visited_cells = np.zeros(n_episodes, dtype=int)

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        ep_len = 0
        ep_min_lidar = float("inf")
        ep_visited = 0
        collided = False
        oob = False

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rews, dones, infos = env.step(action)
            ep_reward += float(rews[0])
            ep_len += 1

            info = infos[0]
            if "min_lidar" in info:
                ep_min_lidar = min(ep_min_lidar, float(info["min_lidar"]))
            if "visited_cells" in info:
                ep_visited = info["visited_cells"]
            if info.get("collision"):
                collided = True
            if info.get("out_of_bounds"):
                oob = True

            if dones[0]:
                break

        rewards[ep] = ep_reward
        lengths[ep] = ep_len
        collisions[ep] = 1 if collided else 0
        oob_terminations[ep] = 1 if oob else 0
        timeout_terminations[ep] = 1 if (not collided and not oob) else 0
        min_lidars[ep] = ep_min_lidar if ep_min_lidar != float("inf") else 0.0
        visited_cells[ep] = ep_visited

        if verbose:
            outcome = "COLL" if collided else ("OOB " if oob else "TIME")
            print(f"  Ep {ep+1:>3d}/{n_episodes}: reward={ep_reward:>8.1f} "
                  f"len={ep_len:>4d} {outcome} cells={ep_visited:>3d} "
                  f"min_lidar={min_lidars[ep]:.2f}")

    env.close()

    # Summary stats
    def stats(arr):
        """Computes mean/std/min/max/median for an array."""
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
        }

    return {
        "model": run_name,
        "n_episodes": n_episodes,
        "reward": stats(rewards),
        "length": stats(lengths),
        "min_lidar": stats(min_lidars),
        "visited_cells": stats(visited_cells),
        "collision_rate": float(np.mean(collisions)),
        "oob_rate": float(np.mean(oob_terminations)),
        "timeout_rate": float(np.mean(timeout_terminations)),
        "raw": {
            "rewards": rewards.tolist(),
            "lengths": lengths.tolist(),
            "collisions": collisions.tolist(),
            "min_lidars": min_lidars.tolist(),
            "visited_cells": visited_cells.tolist(),
        },
    }


def print_summary(results):
    """Prints a formatted summary table for one or more models.

    Args:
        results: A single result dict or a list of result dicts
            as returned by ``eval_single``.
    """
    if not isinstance(results, list):
        results = [results]

    print("\n" + "=" * 92)
    print(f"{'Metric':<20} " + " ".join(f"{r['model']:>20}" for r in results))
    print("-" * 92)

    rows = [
        ("Episodes",        lambda r: f"{r['n_episodes']:>20d}"),
        ("Reward mean",     lambda r: f"{r['reward']['mean']:>20.1f}"),
        ("Reward std",      lambda r: f"{r['reward']['std']:>20.1f}"),
        ("Reward min",      lambda r: f"{r['reward']['min']:>20.1f}"),
        ("Reward max",      lambda r: f"{r['reward']['max']:>20.1f}"),
        ("",                lambda r: ""),
        ("Length mean",     lambda r: f"{r['length']['mean']:>20.1f}"),
        ("Length min",      lambda r: f"{r['length']['min']:>20.0f}"),
        ("Length max",      lambda r: f"{r['length']['max']:>20.0f}"),
        ("",                lambda r: ""),
        ("Collision rate",  lambda r: f"{r['collision_rate']*100:>19.1f}%"),
        ("OOB rate",        lambda r: f"{r['oob_rate']*100:>19.1f}%"),
        ("Timeout rate",    lambda r: f"{r['timeout_rate']*100:>19.1f}%"),
        ("",                lambda r: ""),
        ("Cells mean",      lambda r: f"{r['visited_cells']['mean']:>20.1f}"),
        ("Cells max",       lambda r: f"{r['visited_cells']['max']:>20.0f}"),
        ("",                lambda r: ""),
        ("Min lidar mean",  lambda r: f"{r['min_lidar']['mean']:>20.3f}"),
        ("Min lidar min",   lambda r: f"{r['min_lidar']['min']:>20.3f}"),
    ]

    for name, fmt in rows:
        print(f"{name:<20} " + " ".join(fmt(r) for r in results))

    print("=" * 92)
    print()
    print("Reading the numbers:")
    print("  * High reward std  = policy is unreliable (runs vary a lot)")
    print("  * High OOB rate    = policy often flies out of bounds (altitude or edges)")
    print("  * High collision   = policy crashes often")
    print("  * High timeout     = policy survives to max_steps (good!)")
    print("  * Low min lidar    = policy threads tight gaps (skill, or risk)")
    print()


def main():
    """Entry point: parse CLI args and run evaluation."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/drone_ppo_final",
                   help="Model path without .zip")
    p.add_argument("--episodes", type=int, default=20,
                   help="Number of eval episodes per model")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for spawn RNG (same seed = same spawns)")
    p.add_argument("--compare", nargs="+", default=None,
                   help="Evaluate multiple models side-by-side")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-episode output")
    args = p.parse_args()

    models_to_eval = args.compare if args.compare else [args.model]

    results = []
    for m in models_to_eval:
        if not os.path.exists(m + ".zip"):
            print(f"Skipping: no such model: {m}.zip")
            continue
        result = eval_single(m, n_episodes=args.episodes, seed=args.seed,
                             verbose=not args.quiet)
        results.append(result)

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
