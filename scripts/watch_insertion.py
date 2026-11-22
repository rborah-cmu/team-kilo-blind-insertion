"""Replay baseline and learned insertion on identical starts.

On macOS use sh scripts/watch_insertion.sh.
"""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch
from stable_baselines3 import PPO
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kilo.config import load_config
from kilo.insertion.env import PegInsertionEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('artifacts/insertion_v4'))
    parser.add_argument('--seed', type=int, default=40000)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--mode', choices=['compare', 'baseline', 'policy'], default='compare')
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--headless', action='store_true', help='check replay without opening a window')
    args = parser.parse_args()
    if args.speed <= 0 or args.episodes < 1:
        parser.error('speed and episodes must be positive')
    torch.set_num_threads(1)
    cfg = load_config(args.run_dir / 'config.yaml')
    model = PPO.load(args.run_dir / 'best_model.zip', device='cpu') if args.mode != 'baseline' else None
    env = PegInsertionEnv(cfg)
    modes = ['baseline', 'policy'] if args.mode == 'compare' else [args.mode]
    window = None
    try:
        for episode in range(args.episodes):
            for mode in modes:
                seed = args.seed + episode
                obs, initial = env.reset(seed=seed)
                if not args.headless and window is None:
                    from mujoco import viewer
                    window = viewer.launch_passive(env.model, env.data)
                if window is not None:
                    window.cam.lookat[:] = np.r_[env.true_hole_xy, env.top_z + .02]
                    window.cam.distance = .27
                    window.cam.azimuth = 120
                    window.cam.elevation = -25
                print(f'\n{mode.upper()} | seed {seed} | estimate error {np.round(initial["estimate_error_xy"]*1000,2)} mm', flush=True)
                for step in range(env.max_steps):
                    start = time.monotonic()
                    if window is not None and not window.is_running():
                        return
                    obs['privileged'][:] = 0
                    action = np.zeros(3) if mode == 'baseline' else model.predict(obs, deterministic=True)[0]
                    obs, _, terminated, truncated, info = env.step(action)
                    if window is not None:
                        if hasattr(window, 'set_texts'):
                            window.set_texts([(None, None, f'{mode.upper()} | seed {seed}',
                                               f'{(step+1)/env.policy_hz:.1f}s | depth {info["depth"]*1000:.1f} mm')])
                        window.sync()
                        time.sleep(max(0., 1/env.policy_hz/args.speed - (time.monotonic()-start)))
                    if terminated or truncated:
                        break
                print(f'Success: {info["success"]} | time {(step+1)/env.policy_hz:.2f}s | peak load {info["peak_load_n"]:.1f} N', flush=True)
                if window is not None:
                    until = time.monotonic() + 1.5
                    while window.is_running() and time.monotonic() < until:
                        window.sync()
                        time.sleep(.02)
    finally:
        if window is not None:
            window.close()
        env.close()


if __name__ == '__main__':
    main()
