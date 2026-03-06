"""Run balance env with fixed seed for reproducibility checks.

Use to verify that fork-equivalent parameters yield the same reward sequence
(same config + seed => same trajectory). Run twice with same --run-id and
--seed and compare output or rewards.

Usage:
  uv run python scripts/debug_balance_repro.py --run-id fork_match --seed 42 --num-steps 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TASK_ID = "MjlabMyoSuite-Balance-Flat-MyoLegsTorso"


def _register_tasks() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src = project_root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import myosuite_mjlab.tasks  # noqa: F401


def _config_fingerprint(env_cfg) -> dict:
    """Extract key params that differ between current and fork-equivalent."""
    obs = getattr(env_cfg, "observations", None) or {}
    actor = obs.get("actor") if isinstance(obs, dict) else getattr(obs, "actor", None)
    actor_terms = getattr(actor, "terms", {}) if actor else {}
    rewards = getattr(env_cfg, "rewards", None) or {}
    curriculum = getattr(env_cfg, "curriculum", None) or {}
    push_vel = curriculum.get("push_vel") if isinstance(curriculum, dict) else None
    stages = []
    if push_vel is not None:
        params = getattr(push_vel, "params", None)
        if isinstance(params, dict):
            stages = params.get("stages", [])
        elif params is not None:
            stages = getattr(params, "stages", [])
    bod = rewards.get("body_ang_vel") if isinstance(rewards, dict) else None
    return {
        "has_com_pos_actor": "com_pos" in actor_terms,
        "has_com_lin_vel_actor": "com_lin_vel" in actor_terms,
        "body_ang_vel_weight": getattr(bod, "weight", None),
        "has_metabolic_cost": "metabolic_cost" in rewards,
        "push_vel_stages_count": len(stages),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default="repro_run",
        help="Label for this run (e.g. current, fork_match)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print final mean reward",
    )
    args = parser.parse_args()

    _register_tasks()

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(TASK_ID)
    env_cfg.scene.num_envs = args.num_envs

    fingerprint = _config_fingerprint(env_cfg)
    if not args.quiet:
        print(f"config_fingerprint {args.run_id}: {fingerprint}")

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    env.reset(seed=args.seed)
    action_dtype = env.action_space.dtype
    if isinstance(action_dtype, str):
        action_dtype = getattr(torch, action_dtype, torch.float32)
    zero_action = torch.zeros(
        env.action_space.shape, device=env.device, dtype=action_dtype
    )

    rewards_list = []
    for _ in range(args.num_steps):
        _, rew, _, _, _ = env.step(zero_action)
        rewards_list.append(float(rew.mean().item()))

    env.close()

    mean_reward = sum(rewards_list) / len(rewards_list) if rewards_list else 0.0
    if args.quiet:
        print(mean_reward)
    else:
        print(f"steps={args.num_steps} seed={args.seed} mean_reward={mean_reward}")
        print(
            f"  step 0 reward={rewards_list[0]!r} step -1 reward={rewards_list[-1]!r}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
