"""Minimal mjlab + myosuite_mjlab repro for rsl-rl scalar std behavior."""

from __future__ import annotations

import argparse

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import myosuite_mjlab.tasks  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="MjlabMyoSuite-Velocity-Flat-MyoLeg")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_cfg = load_env_cfg(args.task)
    env_cfg.num_envs = args.num_envs
    env_cfg.device = args.device

    rl_cfg = load_rl_cfg(args.task)
    rl_cfg.max_iterations = args.iters
    rl_cfg.actor.noise_std_type = "scalar"

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    vec_env = RslRlVecEnvWrapper(env)

    runner_cls = load_runner_cls(args.task)
    if runner_cls is None:
        raise RuntimeError(f"Task {args.task} did not register a runner class.")

    runner = runner_cls(vec_env, rl_cfg, log_dir=None, device=args.device)
    runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)

    vec_env.close()


if __name__ == "__main__":
    main()
