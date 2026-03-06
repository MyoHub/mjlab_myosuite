"""Full convergence training script for MyoLegsTorso balance task.

Runs 30,000 PPO iterations with 4096 parallel environments by default.
Mirrors scripts/repro_rsl_rl_std.py but with balance-appropriate defaults,
a persistent log directory, and optional W&B + video support.

Usage:
    # GPU full run (recommended):
    uv run python scripts/train_balance.py

    # W&B logging only:
    uv run python scripts/train_balance.py --wandb --wandb-project my-project

    # W&B + video every 1000 iters:
    uv run python scripts/train_balance.py --wandb --video-interval 1000

    # CPU debug (4 envs):
    uv run python scripts/train_balance.py --num-envs 4 --device cpu

Convergence criteria (monitor in tensorboard/wandb):
    - Mean episode length >= 18 s  (of 20 s max)
    - Mean episode return plateau   (<5% variation over 500 iterations)
    - Action scale = 1.0            (reached at ~iter 12,500)
    - Push scale = 1.0              (reached at ~iter 27,100)
"""

from __future__ import annotations

import argparse
import dataclasses
import os

import numpy as np

TASK = "MjlabMyoSuite-Balance-Flat-MyoLegsTorso"
_VIDEO_STEPS = 200  # env steps per recorded video clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4096,
        help="Number of parallel simulation environments (default: 4096)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device (default: cuda:0)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/balance",
        help="Directory for checkpoints and tensorboard logs (default: logs/balance)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        default="mjlab-balance",
        help="W&B project name (default: mjlab-balance)",
    )
    parser.add_argument(
        "--video-interval",
        type=int,
        default=0,
        help=(
            "Upload a policy rollout video to W&B every N training iterations "
            "(0 = disabled, requires --wandb)"
        ),
    )
    return parser.parse_args()


def _try_record_video(runner: object, n_steps: int = _VIDEO_STEPS) -> np.ndarray | None:
    """Roll out the current policy and collect rendered frames.

    Returns a uint8 (T, H, W, 3) array or None if rendering is not supported.
    Errors are caught silently so a missing renderer never blocks training.
    """
    import torch

    try:
        # Get noise-free inference policy
        if hasattr(runner.alg, "get_inference_policy"):
            policy = runner.alg.get_inference_policy(device=runner.device)
        else:
            policy = runner.alg.actor.act_inference

        obs, _ = runner.env.reset()
        frames = []
        for _ in range(n_steps):
            # runner.env is RslRlVecEnvWrapper; .env is ManagerBasedRlEnv
            frame = runner.env.env.render()
            if frame is None:
                return None
            frames.append(np.asarray(frame, dtype=np.uint8))
            with torch.no_grad():
                result = policy(obs)
                actions = result[0] if isinstance(result, (tuple, list)) else result
            obs = runner.env.step(actions)[0]
        return np.stack(frames) if frames else None
    except Exception:
        return None


def main() -> None:
    if not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "egl"
    if not os.environ.get("MUJOCO_EGL_DEVICE_ID"):
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    import myosuite_mjlab.tasks  # noqa: F401

    args = parse_args()

    env_cfg = load_env_cfg(TASK)
    env_cfg.num_envs = args.num_envs
    env_cfg.device = args.device

    rl_cfg = load_rl_cfg(TASK)

    runner_cls = load_runner_cls(TASK)
    if runner_cls is None:
        raise RuntimeError(f"No runner registered for task: {TASK}")

    # Build runner config dict; add W&B keys when requested.
    train_cfg = dataclasses.asdict(rl_cfg)
    if args.wandb:
        train_cfg["logger"] = "wandb"
        train_cfg["wandb_project"] = args.wandb_project

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    vec_env = RslRlVecEnvWrapper(env)
    runner = runner_cls(vec_env, train_cfg, log_dir=args.log_dir, device=args.device)

    total = rl_cfg.max_iterations

    if args.wandb and args.video_interval > 0:
        # Split training into chunks so we can upload a video at each checkpoint.
        # rsl-rl learn() treats num_learning_iterations as *additional* iterations
        # on top of runner.current_learning_iteration, so chunking works naturally.
        import wandb

        done = 0
        first_chunk = True
        while done < total:
            chunk = min(args.video_interval, total - done)
            runner.learn(
                num_learning_iterations=chunk,
                init_at_random_ep_len=first_chunk,
            )
            first_chunk = False
            done += chunk

            frames = _try_record_video(runner)
            if frames is not None:
                wandb.log(
                    {"video/policy": wandb.Video(frames, fps=30, format="mp4")},
                    step=done,
                )
    else:
        runner.learn(num_learning_iterations=total, init_at_random_ep_len=True)

    vec_env.close()


if __name__ == "__main__":
    main()
