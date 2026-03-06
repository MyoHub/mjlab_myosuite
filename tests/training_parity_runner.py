"""Run N env steps with fixed seed and zero action; print rewards as JSON.

Used by test_training_matches_mjlab_fork to compare our env vs fork env.
Usage:
  uv run python -m tests.training_parity_runner --task TASK_ID --seed 42 --num-steps 10
  uv run python -m tests.training_parity_runner --task TASK_ID --seed 42 --num-steps 10 --fork /path/to/mjlab/src
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Defaults for minimal run
NUM_ENVS = 4
DEVICE = "cpu"


def _register_fork_tasks(fork_src: Path) -> None:
    """Import fork's musculoskeletal task configs so registry has fork's env_cfgs."""
    sys.path.insert(0, str(fork_src))
    import mjlab.tasks.balance.config.myolegtorso  # noqa: F401
    import mjlab.tasks.velocity.config.myoleg  # noqa: F401
    import mjlab.tasks.velocity.config.myolegtorso  # noqa: F401
    import mjlab.tasks.velocity.config.myoskeleton  # noqa: F401


def _register_our_tasks() -> None:
    """Import myosuite_mjlab tasks so registry has our env_cfgs."""
    import myosuite_mjlab.tasks  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        help="Task ID (e.g. MjlabMyoSuite-Balance-Flat-MyoLegsTorso)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument(
        "--fork", type=Path, default=None, help="Path to mjlab fork src/ dir"
    )
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    args = parser.parse_args()

    if args.fork is not None:
        args.fork = args.fork.resolve()
        if not (args.fork / "mjlab").exists():
            print(
                json.dumps({"error": f"Fork path has no mjlab dir: {args.fork}"}),
                file=sys.stderr,
            )
            return 1
        _register_fork_tasks(args.fork)
    else:
        # Ensure our package is importable (project root or installed)
        project_root = Path(__file__).resolve().parents[1]
        src = project_root / "src"
        if src.exists() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
        _register_our_tasks()

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs

    env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE)
    env.reset(seed=args.seed)
    rewards: list[list[float]] = []
    action_dtype = env.action_space.dtype
    if isinstance(action_dtype, str):
        action_dtype = getattr(torch, action_dtype, torch.float32)
    for _ in range(args.num_steps):
        action = torch.zeros(
            env.action_space.shape, device=env.device, dtype=action_dtype
        )
        obs, rew, term, trunc, info = env.step(action)
        # rew: (num_envs,) tensor
        r = rew.cpu().tolist()
        rewards.append(r if isinstance(r, list) else [r])
    env.close()

    print(
        json.dumps(
            {"rewards": rewards, "num_envs": args.num_envs, "num_steps": args.num_steps}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
