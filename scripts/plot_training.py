"""Plot or watch PPO episode returns from Stable-Baselines3 monitor files."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_episodes(log_dir: Path) -> list[tuple[float, float, int, float | None, float | None]]:
    episodes = []
    for path in sorted(log_dir.glob("*.monitor.csv")):
        with path.open(newline="") as stream:
            rows = csv.DictReader(line for line in stream if not line.startswith("#"))
            for row in rows:
                success = row.get("success")
                if success in ("True", "true", "1"):
                    success_value = 1.0
                elif success in ("False", "false", "0"):
                    success_value = 0.0
                else:
                    success_value = None
                noise = row.get("estimate_noise_xy")
                episodes.append((float(row["t"]), float(row["r"]), int(row["l"]),
                                 success_value, float(noise) if noise not in (None, "") else None))
    episodes.sort(key=lambda episode: episode[0])
    return episodes


def draw(log_dir: Path, output: Path, window: int) -> None:
    episodes = read_episodes(log_dir)
    if not episodes:
        print(f"No completed episodes found in {log_dir}")
        return

    returns = np.asarray([episode[1] for episode in episodes])
    lengths = np.asarray([episode[2] for episode in episodes])
    x = np.arange(1, len(episodes) + 1)
    kernel = np.ones(min(window, len(episodes))) / min(window, len(episodes))
    rolling = np.convolve(returns, kernel, mode="valid")
    rolling_x = x[len(x) - len(rolling):]

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), layout="constrained")
    axes[0].plot(x, returns, color="#9db7d5", linewidth=0.7, label="Episode return")
    axes[0].plot(rolling_x, rolling, color="#163e66", linewidth=2, label=f"{window}-episode mean")
    axes[0].set_ylabel("Return")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.2)
    success_recorded = any(episode[3] is not None for episode in episodes)
    title = "PPO insertion training" if success_recorded else "PPO insertion training — success flag not logged"
    axes[0].set_title(title)
    axes[1].plot(x, lengths, color="#d49a53", linewidth=0.8)
    axes[1].set_ylabel("Episode steps")
    axes[1].grid(alpha=0.2)
    successes = [episode[3] for episode in episodes]
    observed = [value for value in successes if value is not None]
    if observed:
        success_x = np.arange(1, len(observed) + 1)
        success_window = min(window, len(observed))
        success_mean = np.convolve(observed, np.ones(success_window) / success_window, mode="valid")
        axes[2].plot(success_x[-len(success_mean):], success_mean, color="#27834a", linewidth=2)
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_ylabel(f"Success rate\n({success_window}-ep mean)")
    else:
        axes[2].text(0.5, 0.5, "Success was not recorded for this run", ha="center", va="center",
                     transform=axes[2].transAxes, color="#666666")
        axes[2].set_yticks([])
        axes[2].set_ylabel("Success rate")
    axes[2].set_xlabel("Completed episode (merged across environments)")
    axes[2].grid(alpha=0.2)
    evaluations = []
    for path in log_dir.parent.glob('evaluation_*.json'):
        try:
            evaluations.append(json.loads(path.read_text()))
        except json.JSONDecodeError:  # A running evaluator may still be writing.
            continue
    evaluations.sort(key=lambda row: row['timesteps'])
    noise_rows = [episode for episode in episodes if episode[3] is not None and episode[4] is not None]
    if evaluations:
        axes[3].plot([row['timesteps'] for row in evaluations],
                     [row['success_rate'] for row in evaluations], marker='o', color='#578c67')
        axes[3].set_ylim(0, 1.05)
        axes[3].set_ylabel('Fixed validation\nsuccess rate')
        axes[3].set_xlabel('Training steps (step 0 is the unlearned reference)')
    elif noise_rows:
        edges = np.arange(0.0, 0.00551, 0.0005)
        rates, labels = [], []
        for lower, upper in zip(edges[:-1], edges[1:]):
            bucket = [episode[3] for episode in noise_rows if lower <= episode[4] < upper]
            if bucket:
                rates.append(sum(bucket) / len(bucket))
                labels.append(f"{lower*1000:.1f}–{upper*1000:.1f}")
        axes[3].bar(labels, rates, color="#578c67")
        axes[3].set_ylim(0, 1.05)
        axes[3].set_ylabel("Success rate")
        axes[3].set_xlabel("Episode noise halfwidth (mm); per-axis uniform range")
    else:
        axes[3].text(0.5, 0.5, "Localization noise was not recorded for this run", ha="center", va="center",
                     transform=axes[3].transAxes, color="#666666")
        axes[3].set_yticks([])
        axes[3].set_ylabel("Success by difficulty")
    axes[3].grid(axis="y", alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    tail = returns[-min(window, len(returns)):]
    success_summary = f"successes: {int(sum(observed))}/{len(observed)}" if observed else "successes: not logged"
    print(f"Episodes: {len(episodes)} | return mean: {returns.mean():.2f} | "
          f"last {len(tail)} mean: {tail.mean():.2f} | {success_summary} | output: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/insertion_v4/monitor"))
    parser.add_argument("--output", type=Path, default=Path("runs/insertion_v4/training_progress.png"))
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--watch", action="store_true", help="refresh the image every 15 seconds")
    args = parser.parse_args()
    while True:
        draw(args.log_dir, args.output, max(1, args.window))
        if not args.watch:
            break
        time.sleep(15)


if __name__ == "__main__":
    main()
