"""Evaluate a saved policy or zero-residual reference on a fixed seed set."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from stable_baselines3 import PPO
from kilo.config import DEFAULT_CONFIG, load_config
from kilo.insertion.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--config', type=Path, help='defaults to the model run config, otherwise configs/default.yaml')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--output', type=Path, default=Path('runs/diagnostics/reference_evaluation.json'))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    torch.set_num_threads(1)
    model = PPO.load(args.model, device='cpu') if args.model else None
    config_path = args.config
    if config_path is None and args.model is not None:
        run_dir = args.model.parent.parent if args.model.parent.name == 'checkpoints' else args.model.parent
        config_path = run_dir / 'config.yaml'
        if not config_path.exists():
            parser.error('No saved config found beside this model; specify --config explicitly.')
    result = evaluate(load_config(config_path or DEFAULT_CONFIG), model, args.episodes, args.seed,
                      progress=lambda done, successes: print(
                          f'\rEvaluation {done}/{args.episodes} | successes {successes}/{done}', end='', flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print('\n' + json.dumps({k:v for k,v in result.items() if k not in ('records', 'configuration')}, indent=2))


if __name__ == '__main__':
    main()
