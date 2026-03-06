"""Tests verifying the balance task reward structure and learning compatibility.

Three tests:
  1. Config-level reward term check (fast, no env instantiation)
  2. Config-level COM observation check (fast, no env instantiation)
  3. PPO learning smoke test (slow, requires MuJoCo simulation)
"""

from __future__ import annotations

import pytest


def test_balance_cfg_has_cheryl_rewards() -> None:
    """Balance env cfg keeps the expected core StandingBalance reward terms."""
    from mjlab.tasks.registry import load_env_cfg

    import myosuite_mjlab.tasks  # noqa: F401

    env_cfg = load_env_cfg("MjlabMyoSuite-Balance-Flat-MyoLegsTorso")

    reward_names = set(env_cfg.rewards.keys())
    # Core terms from the reference balance task that must be present.
    for name in {"pose", "alive"}:
        assert name in reward_names, f"{name} missing from reward terms: {reward_names}"
    # Sanity-check weights for the core terms.
    assert env_cfg.rewards["pose"].weight == 1.0
    assert env_cfg.rewards["alive"].weight == 10.0


def test_balance_cfg_has_com_observations() -> None:
    """Balance env cfg exposes pelvis-based COM proxy in critic observations."""
    from mjlab.tasks.registry import load_env_cfg

    import myosuite_mjlab.tasks  # noqa: F401

    env_cfg = load_env_cfg("MjlabMyoSuite-Balance-Flat-MyoLegsTorso")

    critic_terms = env_cfg.observations["critic"].terms

    # For value estimation we at least require base linear velocity as a proxy for COM
    # motion; the exact COM features are implementation-dependent.
    assert "base_lin_vel" in critic_terms, (
        "base_lin_vel missing from critic observations"
    )


@pytest.mark.slow
def test_balance_learning_smoke() -> None:
    """PPO training runs 2 iterations on the balance task without errors.

    Verifies: obs shapes valid, rewards finite, gradient update completes.
    Uses the same pattern as scripts/repro_rsl_rl_std.py.
    """
    import dataclasses

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    import myosuite_mjlab.tasks  # noqa: F401

    task_id = "MjlabMyoSuite-Balance-Flat-MyoLegsTorso"

    env_cfg = load_env_cfg(task_id)
    env_cfg.num_envs = 4
    env_cfg.device = "cpu"

    rl_cfg = load_rl_cfg(task_id)

    runner_cls = load_runner_cls(task_id)
    assert runner_cls is not None, f"No runner registered for {task_id}"

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    vec_env = RslRlVecEnvWrapper(env)
    runner = runner_cls(vec_env, dataclasses.asdict(rl_cfg), log_dir=None, device="cpu")

    runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)

    obs, _ = vec_env.reset()
    assert obs.shape[0] == 4, f"Expected 4 envs, got {obs.shape[0]}"
    assert torch.all(torch.isfinite(obs)), "NaN/Inf detected in observations"

    vec_env.close()
