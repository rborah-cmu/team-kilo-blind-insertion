"""Matched-seed evaluation at a fixed localization error distribution."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from kilo.insertion.env import PegInsertionEnv


def evaluate(config, model=None, episodes=20, seed=10000, progress=None):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    env = PegInsertionEnv(config)
    records = []
    try:
        for trial in range(episodes):
            obs, initial = env.reset(seed=seed + trial)
            episode_return = 0.0
            for step in range(env.max_steps):
                # Deployment needs no privileged inputs. The actor is tested
                # with zero placeholders for the training-only critic state.
                obs['privileged'] = np.zeros_like(obs['privileged'])
                action = np.zeros(3) if model is None else model.predict(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            records.append({
                'seed': seed + trial, 'success': bool(info['success']),
                'seconds': (step + 1) / env.policy_hz,
                'error_xy_mm': (initial['estimate_error_xy'] * 1000).tolist(),
                'return': episode_return, 'depth_mm': info['depth'] * 1000,
                'entry_error_mm': info['entry_error'] * 1000,
                'peak_load_n': float(info['peak_load_n']),
                'dropped': info['dropped'], 'peak_grasp_slip_mm': 1000*info['peak_grasp_slip_m'],
                'reward_terms': {k:float(v) for k,v in info.items() if k.startswith('reward_')},
            })
            if progress is not None:
                progress(trial + 1, sum(row['success'] for row in records))
    finally:
        env.close()
    success = [r for r in records if r['success']]
    bins = defaultdict(list)
    for row in records:
        radius = np.linalg.norm(row['error_xy_mm'])
        bins['0–2 mm' if radius < 2 else '2–4 mm' if radius < 4 else '>4 mm'].append(row['success'])
    return {
        'episodes': episodes, 'successes': len(success), 'success_rate': len(success)/episodes,
        'mean_success_seconds': float(np.mean([r['seconds'] for r in success])) if success else None,
        'mean_return': float(np.mean([r['return'] for r in records])),
        'max_peak_load_n': max(row['peak_load_n'] for row in records),
        'controller': 'zero_residual' if model is None else 'deterministic_policy',
        'noise_halfwidth_mm': float(config['insertion']['train_noise_xy']) * 1000,
        'max_episode_seconds': env.max_steps / env.policy_hz,
        'configuration': config,
        'success_by_actual_error': {k:{'count':len(v),'successes':sum(v)} for k,v in bins.items()},
        'records': records,
    }
