"""Train the estimate-guided insertion policy with PPO on CPU.

Example: python scripts/train_insertion.py --timesteps 2000000 --envs 8
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import hashlib
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import json
import yaml

# MuJoCo environments do small linear-algebra operations. Keep each worker's
# BLAS runtime single-threaded to avoid multiplying threads across workers.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# Make direct execution from the repository root (or another directory) find
# the local ``kilo`` package without requiring an editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kilo.config import DEFAULT_CONFIG, load_config
from kilo.insertion.env import FRAME_FEATURES, PegInsertionEnv
from kilo.insertion.policy import AsymmetricInsertionPolicy
from kilo.insertion.evaluation import evaluate
from stable_baselines3.common.logger import configure


class TerminalProgressCallback(BaseCallback):
    """Compact single-line progress display, without SB3's periodic tables."""

    def __init__(self, total_steps: int, refresh_seconds: float = 0.5):
        super().__init__(verbose=0)
        self.total_steps = max(1, int(total_steps))
        self.refresh_seconds = refresh_seconds
        self.start_time = 0.0
        self.last_refresh = 0.0
        self.start_num_timesteps = 0

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:d}:{seconds:02d}"

    def _on_training_start(self) -> None:
        self.start_time = self.last_refresh = time.monotonic()
        self.start_num_timesteps = self.model.num_timesteps
        self._render(force=True)

    def _render(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_refresh < self.refresh_seconds:
            return
        done = min(self.total_steps, max(0, self.model.num_timesteps - self.start_num_timesteps))
        elapsed = now - self.start_time
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (self.total_steps - done) / rate if rate > 0 else 0.0
        width = 28
        filled = round(width * done / self.total_steps)
        bar = "█" * filled + "░" * (width - filled)
        percent = 100.0 * done / self.total_steps
        line = (f"PPO [{bar}] {percent:5.1f}%  {done:,}/{self.total_steps:,} steps"
                f" | {rate:,.0f} steps/s | elapsed {self._duration(elapsed)}"
                f" | ETA {self._duration(remaining)}")
        sys.stdout.write("\r" + line)
        sys.stdout.flush()
        self.last_refresh = now

    def _on_step(self) -> bool:
        self._render()
        return True

    def _on_training_end(self) -> None:
        self._render(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()


class FixedEvaluationCallback(BaseCallback):
    """Evaluate on the same unseen seeds; choose checkpoints by task success."""
    def __init__(self, config, output_dir, interval, episodes):
        super().__init__()
        self.config, self.output_dir = config, output_dir
        self.interval, self.episodes = interval, episodes
        self.next_eval = 0
        self.best_score = (-1.0, -float('inf'))

    def _evaluate(self):
        width = self.config['insertion']['train_noise_xy'] * 1000
        print(f"\nEvaluating {self.episodes} fixed starts at +/-{width:g} mm...", flush=True)
        result = evaluate(self.config, self.model, self.episodes, seed=10000,
                          progress=lambda done, successes: print(
                              f"\rValidation {done}/{self.episodes} | successes {successes}/{done}",
                              end='', flush=True))
        result['timesteps'] = self.model.num_timesteps
        path = self.output_dir / f"evaluation_{self.model.num_timesteps}.json"
        path.write_text(json.dumps(result, indent=2))
        seconds = result['mean_success_seconds']
        score = (result['success_rate'], -seconds if seconds is not None else -float('inf'))
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.output_dir / 'best_model')
        print(f"\nEvaluation: {result['successes']}/{self.episodes} successful; mean successful time {seconds}", flush=True)

    def _on_training_start(self):
        self._evaluate()
        self.next_eval = self.model.num_timesteps + self.interval

    def _on_step(self):
        if self.model.num_timesteps >= self.next_eval:
            self._evaluate()
            self.next_eval = self.model.num_timesteps + self.interval
        return True

    def _on_training_end(self):
        self._evaluate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/insertion_v4"))
    parser.add_argument("--vec-env", choices=("subproc", "dummy"), default="subproc")
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume from an SB3 checkpoint and train to the requested total timesteps")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ins = cfg["insertion"]
    default_envs = min(int(ins["num_envs"]), max(1, (os.cpu_count() or 2) - 2))
    n_envs = args.envs if args.envs is not None else default_envs
    total_timesteps = args.timesteps if args.timesteps is not None else int(ins["total_timesteps"])
    if n_envs < 1 or total_timesteps < 1:
        parser.error('--envs and --timesteps must be positive')
    output_dir = args.output_dir
    if (output_dir / 'config.yaml').exists():
        raise FileExistsError(f"{output_dir} already contains a run. Choose a new --output-dir.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    sources = ['kilo/insertion/env.py', 'kilo/insertion/reference.py', 'kilo/insertion/policy.py',
               'kilo/insertion/evaluation.py', 'kilo/sim/scene.py', 'scripts/train_insertion.py']
    manifest = {'seed': args.seed, 'envs': n_envs, 'timesteps': total_timesteps,
                'resume': str(args.resume) if args.resume else None,
                'versions': {package: version(package) for package in ('mujoco', 'torch', 'stable-baselines3', 'gymnasium')},
                'source_hashes': {path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() for path in sources}}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    torch.set_num_threads(1)

    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else None
    vec_env_kwargs = {}
    if args.vec_env == "subproc":
        vec_env_kwargs["start_method"] = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
    env = make_vec_env(
        PegInsertionEnv,
        n_envs=n_envs,
        seed=args.seed,
        env_kwargs={"config": cfg, "use_tau_ext": True, "residual": True},
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_dir=str(output_dir / "monitor"),
        monitor_kwargs={"info_keywords": ("success", "estimate_noise_xy", "depth", "entry_error", "peak_load_n", "dropped", "peak_grasp_slip_m",
                                         "reward_shaping", "reward_time", "reward_force", "reward_smoothness", "reward_success", "reward_failure")},
    )
    checkpoint = CheckpointCallback(
        save_freq=max(1, 100_000 // n_envs),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_insertion",
    )
    model = None
    try:
        if args.resume is not None:
            model = PPO.load(args.resume, device="cpu")
            if not isinstance(model.policy, AsymmetricInsertionPolicy):
                raise ValueError('This checkpoint predates the new observations/controller. Start v4 fresh.')
            if not math.isclose(model.gamma, float(ins['ppo_gamma'])):
                raise ValueError('Checkpoint gamma must match ppo_gamma for potential-based shaping.')
            model.set_env(env)
            remaining = max(0, total_timesteps - model.num_timesteps)
            print(f"Resuming at {model.num_timesteps:,} steps; target {total_timesteps:,} "
                  f"({remaining:,} remaining).")
        else:
            remaining = total_timesteps
            model = PPO(
                AsymmetricInsertionPolicy,
                env,
                learning_rate=1e-4,
                n_steps=int(ins["ppo_n_steps"]),
                batch_size=int(ins["ppo_batch_size"]),
                gamma=float(ins["ppo_gamma"]),
                gae_lambda=float(ins["ppo_gae_lambda"]),
                clip_range=0.2,
                ent_coef=0.001,
                n_epochs=5,
                target_kl=0.02,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs={
                    "log_std_init": -2.3,
                    "features_extractor_kwargs": {
                        "history_frames": int(ins["history_frames"]),
                        "frame_size": FRAME_FEATURES,
                        "features_dim": 128,
                    },
                },
                device="cpu",
                seed=args.seed,
                verbose=0,
            )
            # The initial mean policy follows the verified zero-residual reference.
            torch.nn.init.zeros_(model.policy.action_net.weight)
            torch.nn.init.zeros_(model.policy.action_net.bias)
        model.verbose = 0
        model.set_logger(configure(str(output_dir / 'ppo_metrics'), ['csv']))
        if remaining:
            progress = TerminalProgressCallback(total_steps=remaining)
            evaluation = FixedEvaluationCallback(cfg, output_dir, int(ins['eval_interval']), int(ins['eval_episodes']))
            model.learn(total_timesteps=remaining,
                        callback=CallbackList([checkpoint, evaluation, progress]),
                        reset_num_timesteps=args.resume is None)
        model.save(output_dir / "ppo_insertion_final")
    except KeyboardInterrupt:
        if model is not None:
            model.save(output_dir / 'ppo_insertion_interrupted')
            print(f'\nStopped at {model.num_timesteps:,} steps; saved ppo_insertion_interrupted.zip.')
    finally:
        env.close()


if __name__ == "__main__":
    main()
