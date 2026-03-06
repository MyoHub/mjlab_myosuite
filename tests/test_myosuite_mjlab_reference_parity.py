"""Validate myosuite_mjlab config parity with the mjlab fork (read-only reference).

Parity tests run only when the fork is available (sibling ../mjlab or MJLAB_FORK_PATH).
They pass only if mjlab_myosuite results match the fork's results.

Loadability tests run without the fork and ensure registered tasks can be loaded.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


def test_musculoskeletal_tasks_loadable_without_fork() -> None:
    """Load env_cfg and rl_cfg for musculoskeletal task IDs (no fork required)."""
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    # Importing the top-level package should register all MyoSuite tasks.
    import myosuite_mjlab  # noqa: F401

    task_ids = [
        "MjlabMyoSuite-Velocity-Flat-MyoLeg",
        "MjlabMyoSuite-Velocity-Flat-MyoSkeleton",
        "MjlabMyoSuite-Standing-Flat-MyoSkeleton",
        "MjlabMyoSuite-Velocity-Flat-MyoLegsTorso",
        "MjlabMyoSuite-Balance-Flat-MyoLegsTorso",
    ]
    for task_id in task_ids:
        env_cfg = load_env_cfg(task_id)
        rl_cfg = load_rl_cfg(task_id)
        runner_cls = load_runner_cls(task_id)
        assert env_cfg is not None, f"env_cfg missing for {task_id}"
        assert rl_cfg is not None, f"rl_cfg missing for {task_id}"
        assert runner_cls is not None, f"runner_cls missing for {task_id}"


def _find_fork_src() -> Path | None:
    """Find mjlab fork src directory for reference. Does not use installed mjlab."""
    # Env override: path to fork repo root (or to fork src)
    env_path = os.environ.get("MJLAB_FORK_PATH")
    if env_path:
        p = Path(env_path).resolve()
        if (p / "src" / "mjlab").exists():
            return p / "src"
        if (p / "mjlab").exists():
            return p
        return None
    # Sibling repo: same parent as mjlab_myosuite repo root
    sibling_root = Path(__file__).resolve().parents[2] / "mjlab"
    sibling_src = sibling_root / "src"
    if (sibling_src / "mjlab" / "tasks" / "velocity" / "config" / "myoleg").exists():
        return sibling_src
    return None


FORK_SRC = _find_fork_src()
PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"

FORK_REQUIRED = FORK_SRC is None


def _extract_reward_weights(src: str) -> dict[str, str]:
    direct = dict(re.findall(r'cfg\.rewards\["([^"]+)"\]\.weight = ([^\s,]+)', src))
    blocks = re.findall(r'cfg\.rewards\["([^"]+)"\] = RewardTermCfg\(([\s\S]*?)\)', src)
    for name, content in blocks:
        w_match = re.search(r"weight=([^\s,)]+)", content)
        if w_match:
            direct[name] = w_match.group(1)
    return direct


def _extract_reward_std(src: str) -> dict[str, str]:
    direct = dict(
        re.findall(r'cfg\.rewards\["([^"]+)"\]\.params\["std"\] = ([^\s,]+)', src)
    )
    blocks = re.findall(r'cfg\.rewards\["([^"]+)"\] = RewardTermCfg\(([\s\S]*?)\)', src)
    for name, content in blocks:
        s_match = re.search(r'["\']std["\']:\s*([^\s,{}]+)', content)
        if s_match:
            direct[name] = s_match.group(1)
    return direct


@pytest.mark.skipif(
    FORK_REQUIRED,
    reason="mjlab fork not found (set MJLAB_FORK_PATH or clone mjlab beside mjlab_myosuite)",
)
def test_myoleg_env_cfg_matches_reference() -> None:
    ref_path = FORK_SRC / "mjlab/tasks/velocity/config/myoleg/env_cfgs.py"
    new_path = PROJECT_SRC / "myosuite_mjlab/tasks/velocity/myoleg/env_cfgs.py"

    ref_src = ref_path.read_text()
    new_src = new_path.read_text()

    ref_weights = _extract_reward_weights(ref_src)
    new_weights = _extract_reward_weights(new_src)
    for key in (
        "track_linear_velocity",
        "track_angular_velocity",
        "pose",
        "upright",
        "body_ang_vel",
        "action_rate_l2",
        "angular_momentum",
    ):
        assert new_weights[key] == ref_weights.get(key), key

    ref_std = _extract_reward_std(ref_src)
    new_std = _extract_reward_std(new_src)
    for key in ("track_linear_velocity", "track_angular_velocity"):
        assert new_std[key] == ref_std[key], key


@pytest.mark.skipif(
    FORK_REQUIRED,
    reason="mjlab fork not found (set MJLAB_FORK_PATH or clone mjlab beside mjlab_myosuite)",
)
def test_myoskeleton_env_cfg_matches_reference() -> None:
    ref_path = FORK_SRC / "mjlab/tasks/velocity/config/myoskeleton/env_cfgs.py"
    new_path = PROJECT_SRC / "myosuite_mjlab/tasks/velocity/myoskeleton/env_cfgs.py"

    ref_src = ref_path.read_text()
    new_src = new_path.read_text()

    ref_weights = _extract_reward_weights(ref_src)
    new_weights = _extract_reward_weights(new_src)
    for key in ("track_linear_velocity", "pose", "upright", "self_collisions"):
        assert new_weights[key] == ref_weights.get(key), key


@pytest.mark.skipif(
    FORK_REQUIRED,
    reason="mjlab fork not found (set MJLAB_FORK_PATH or clone mjlab beside mjlab_myosuite)",
)
def test_myolegtorso_env_cfg_matches_reference() -> None:
    ref_path = FORK_SRC / "mjlab/tasks/velocity/config/myolegtorso/env_cfgs.py"
    new_path = PROJECT_SRC / "myosuite_mjlab/tasks/velocity/myolegtorso/env_cfgs.py"

    ref_src = ref_path.read_text()
    new_src = new_path.read_text()

    ref_weights = _extract_reward_weights(ref_src)
    new_weights = _extract_reward_weights(new_src)
    for key in ("track_linear_velocity", "pose", "upright", "cyclic_hip"):
        assert new_weights[key] == ref_weights.get(key), key


@pytest.mark.skipif(
    FORK_REQUIRED,
    reason="mjlab fork not found (set MJLAB_FORK_PATH or clone mjlab beside mjlab_myosuite)",
)
def test_myolegtorso_balance_env_cfg_matches_reference() -> None:
    ref_path = FORK_SRC / "mjlab/tasks/balance/config/myolegtorso/env_cfgs.py"
    new_path = (
        PROJECT_SRC / "myosuite_mjlab/tasks/balance/config/myolegtorso/env_cfgs.py"
    )

    ref_src = ref_path.read_text()
    new_src = new_path.read_text()

    ref_weights = _extract_reward_weights(ref_src)
    new_weights = _extract_reward_weights(new_src)
    for key in (
        "upright",
        "upright_torso",
        "pose",
        "body_ang_vel",
        "dof_pos_limits",
        "action_rate_l2",
        "alive",
    ):
        assert new_weights[key] == ref_weights.get(key), key
