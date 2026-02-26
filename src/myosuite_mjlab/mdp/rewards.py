"""Extra MDP reward helpers for musculoskeletal tasks (standalone)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def cyclic_hip_flexion_penalty(
    env: ManagerBasedRlEnv,
    hip_period: int,
    amplitude: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize deviation from a phase-based hip flexion pattern (myoLegWalk-style)."""
    asset: Entity = env.scene[asset_cfg.name]
    if asset_cfg.joint_names is not None:
        joint_ids, _ = asset.find_joints(asset_cfg.joint_names)
        jids = joint_ids
    else:
        jids = asset_cfg.joint_ids
    phase = (env.common_step_counter % hip_period) / hip_period
    two_pi = 2.0 * math.pi
    des_l = amplitude * math.cos(phase * two_pi + math.pi)
    des_r = amplitude * math.cos(phase * two_pi)
    desired = torch.tensor(
        [des_l, des_r],
        device=env.device,
        dtype=asset.data.joint_pos.dtype,
    )
    desired = desired.unsqueeze(0).expand(env.num_envs, 2)
    actual = asset.data.joint_pos[:, jids]
    error = torch.norm(actual - desired, dim=1)
    return error
