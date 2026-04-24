"""Compare metrics across TensorBoard runs in ``logs/``.

Prints a table showing key stats (final values, peaks, wall time)
for every run matching a pattern.  Handy for reviewing what tuning
changes actually moved the needle.

Example output (``--mode final``)::

    === Mode: final ===
       run      steps     time   ep_len   reward expl_var    std     kl clip_frac val_loss  cells  collis
    ---------------------------------------------------------------------------------------------------------------
        v14    500,000  2h 15m    987.3    312.5    0.871  0.412  0.018   0.052    0.123    145      23
        v15    500,000  2h 30m   1204.1    458.2    0.912  0.385  0.015   0.041    0.098    198      11

Usage::

    python compare_runs.py                       # all palmetto runs
    python compare_runs.py --pattern v6 v14 v15  # specific versions
    python compare_runs.py --pattern test        # anything with "test"
    python compare_runs.py --sort-by ep_rew_mean # sort by a metric

The wall-clock time is pulled from event timestamps, not
``time/time_elapsed``, since SB3 does not always log that metric.
"""

import argparse
import os
import re
import sys
from datetime import datetime

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Need tensorboard: pip install tensorboard")
    sys.exit(1)


# Metrics to compare.  Tag -> (short name, format, higher-is-better)
METRICS = [
    ("rollout/ep_len_mean",       "ep_len",        "{:>8.1f}", True),
    ("rollout/ep_rew_mean",       "reward",        "{:>8.1f}", True),
    ("train/explained_variance",  "expl_var",      "{:>8.3f}", True),
    ("train/std",                 "std",           "{:>6.3f}", None),  # lower = committed
    ("train/approx_kl",           "kl",            "{:>6.3f}", None),
    ("train/clip_fraction",       "clip_frac",     "{:>7.3f}", None),
    ("train/value_loss",          "val_loss",      "{:>8.3f}", False),
    ("custom/visited_cells",      "cells",         "{:>6.0f}", True),
    ("custom/collisions",         "collis",        "{:>7.0f}", False),
]


def load_run(log_dir):
    """Loads a single TensorBoard run and extracts key metrics.

    Args:
        log_dir: Path to the TensorBoard event directory.

    Returns:
        A dict with ``"name"``, ``"metrics"``, ``"duration_s"``,
        ``"final_step"``, and ``"started"`` keys.
    """
    ea = EventAccumulator(log_dir)
    ea.Reload()
    tags = set(ea.Tags()["scalars"])

    out = {
        "name": os.path.basename(log_dir).replace("drone_ppo_palmetto_", "").replace("_1", ""),
        "metrics": {},
    }

    # Duration from event timestamps
    ref = "rollout/ep_len_mean"
    if ref in tags:
        scalars = ea.Scalars(ref)
        if scalars:
            out["duration_s"] = scalars[-1].wall_time - scalars[0].wall_time
            out["final_step"] = scalars[-1].step
            out["started"] = datetime.fromtimestamp(scalars[0].wall_time)
        else:
            out["duration_s"] = 0
            out["final_step"] = 0
            out["started"] = None

    for tag, short, _fmt, _hib in METRICS:
        if tag not in tags:
            out["metrics"][short] = None
            continue
        scalars = ea.Scalars(tag)
        if not scalars:
            out["metrics"][short] = None
            continue
        final = scalars[-1].value
        peak = max(scalars, key=lambda s: s.value)
        trough = min(scalars, key=lambda s: s.value)
        out["metrics"][short] = {
            "final": final,
            "peak": peak.value,
            "peak_step": peak.step,
            "trough": trough.value,
            "trough_step": trough.step,
        }
    return out


def format_duration(seconds):
    """Formats a duration in seconds as ``Xh Ym``.

    Args:
        seconds: Duration in seconds, or a falsy value.

    Returns:
        A human-readable string like ``"2h 15m"`` or ``"n/a"``.
    """
    if not seconds:
        return "n/a"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m:2d}m"


def print_table(runs, mode="final"):
    """Prints a formatted comparison table to stdout.

    Args:
        runs: List of run dicts as returned by ``load_run``.
        mode: One of ``"final"`` (last logged value), ``"peak"``
            (best value seen), or ``"drift"`` (peak minus final,
            to spot late-training collapse).
    """
    headers = ["run", "steps", "time"] + [m[1] for m in METRICS]
    widths = [6, 10, 8] + [max(len(m[1]), 8) for m in METRICS]

    # Header
    header_fmt = " ".join(f"{{:>{w}}}" for w in widths)
    print(f"\n=== Mode: {mode} ===")
    print(header_fmt.format(*headers))
    print("-" * (sum(widths) + len(widths) - 1))

    for run in runs:
        row = [run["name"]]
        row.append(f"{run.get('final_step', 0):,}")
        row.append(format_duration(run.get("duration_s")))
        for tag, short, fmt, _hib in METRICS:
            data = run["metrics"].get(short)
            if data is None:
                row.append("n/a")
                continue
            if mode == "final":
                row.append(fmt.format(data["final"]).strip())
            elif mode == "peak":
                row.append(fmt.format(data["peak"]).strip())
            elif mode == "drift":
                drift = data["peak"] - data["final"]
                row.append(fmt.format(drift).strip())
            else:
                row.append("?")
        print(header_fmt.format(*row))


def _version_key(name):
    """Extracts a numeric version from a run name for natural sorting.

    Looks for the first ``vN`` or ``vNN`` pattern in *name* and
    returns the integer.  Falls back to 0 for non-versioned names.

    Args:
        name: Run directory name (e.g. ``"drone_ppo_palmetto_v14_1"``).

    Returns:
        An integer suitable as a sort key.
    """
    match = re.search(r"v(\d+)", name)
    return int(match.group(1)) if match else 0


def main():
    """Entry point: parse args, load runs, and print comparison tables."""
    p = argparse.ArgumentParser()
    p.add_argument("--logs-dir", default="logs", help="Root logs directory")
    p.add_argument("--pattern", nargs="*", default=None,
                   help="Only include runs whose name contains any of these")
    p.add_argument("--sort-by", default=None,
                   help="Sort runs by a metric short-name (ep_len, reward, etc.)")
    p.add_argument("--mode", choices=["final", "peak", "drift", "all"],
                   default="all", help="Which values to show")
    args = p.parse_args()

    if not os.path.isdir(args.logs_dir):
        print(f"No such dir: {args.logs_dir}")
        sys.exit(1)

    # Find palmetto runs by default
    all_dirs = os.listdir(args.logs_dir)
    if args.pattern:
        dirs = [d for d in all_dirs if any(p in d for p in args.pattern)]
    else:
        dirs = [d for d in all_dirs if "palmetto" in d]

    if not dirs:
        print("No runs matched")
        sys.exit(0)

    # Natural sort: put v6 before v10
    dirs.sort(key=_version_key)

    runs = []
    for d in dirs:
        try:
            run = load_run(os.path.join(args.logs_dir, d))
            runs.append(run)
        except Exception as e:
            print(f"Skipping {d}: {e}")

    if args.sort_by:
        runs.sort(
            key=lambda r: (r["metrics"].get(args.sort_by) or {"final": 0})["final"],
            reverse=True,
        )

    if args.mode == "all":
        print_table(runs, "final")
        print_table(runs, "peak")
        print_table(runs, "drift")
        print()
        print("drift = peak - final  (large values = late-training decay)")
    else:
        print_table(runs, args.mode)


if __name__ == "__main__":
    main()
