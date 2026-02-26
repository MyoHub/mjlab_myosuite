"""Extra MDP curriculum helpers for balance/synergy tasks (standalone, no fork)."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class ActionScaleStage(TypedDict):
    step: int
    scale: float


class PushScaleStage(TypedDict):
    step: int
    scale: float


def action_scale_curriculum(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    action_name: str,
    stages: list[ActionScaleStage],
) -> torch.Tensor:
    """Ramp action scale over training (e.g. 0.05 at start, 1.0 later)."""
    del env_ids
    step = env.common_step_counter
    scale = 0.0
    for s in stages:
        if step >= s["step"]:
            scale = s["scale"]
    action_cfg = env.action_manager.cfg.get(action_name)
    if action_cfg is not None and hasattr(action_cfg, "scale"):
        action_cfg.scale = scale
    return torch.tensor([scale], device=env.device)


def push_velocity_curriculum(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    event_name: str,
    base_velocity_range: dict[str, tuple[float, float]],
    stages: list[PushScaleStage],
) -> torch.Tensor:
    """Ramp push disturbance strength over training (0 = no push, 1 = full)."""
    del env_ids
    step = env.common_step_counter
    scale = 0.0
    for s in stages:
        if step >= s["step"]:
            scale = s["scale"]
    event_cfg = env.event_manager.get_term_cfg(event_name)
    scaled = {
        k: (lo * scale, hi * scale) for k, (lo, hi) in base_velocity_range.items()
    }
    event_cfg.params["velocity_range"] = scaled
    return torch.tensor([scale], device=env.device)
