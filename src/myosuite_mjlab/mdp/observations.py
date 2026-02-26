"""Extra MDP observation helpers for musculoskeletal models (standalone, no fork).

Provides tendon/COM observations used by velocity and balance tasks when
public mjlab's tasks.velocity.mdp does not export them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def tendon_length(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Tendon lengths (muscle length proxy). Shape (num_envs, num_tendons)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.tendon_len


def tendon_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Tendon velocities (muscle velocity proxy). Shape (num_envs, num_tendons)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.tendon_vel


def actuator_force(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Actuator forces (muscle force). Shape (num_envs, num_actuators)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.actuator_force


def com_pos_w(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Full-body COM position in world frame. Shape (num_envs, 3)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.data.subtree_com[:, asset.data.indexing.root_body_id].clone()


def com_lin_vel_w(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Full-body COM linear velocity in world frame. Shape (num_envs, 3)."""
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset.data.indexing.body_ids
    mass_arr = asset.data.model.body_mass
    if hasattr(mass_arr, "__getitem__") and mass_arr.ndim >= 1:
        bid = body_ids.cpu().numpy() if isinstance(body_ids, torch.Tensor) else body_ids
        mass = mass_arr[0, bid] if mass_arr.ndim == 2 else mass_arr[bid]
    else:
        mass = mass_arr
    if not isinstance(mass, torch.Tensor):
        mass = torch.as_tensor(
            mass,
            device=asset.data.body_com_lin_vel_w.device,
            dtype=torch.float32,
        )
    vel = asset.data.body_com_lin_vel_w  # (num_envs, num_bodies, 3)
    mass_2d = mass.unsqueeze(0).unsqueeze(-1)  # (1, num_bodies, 1)
    total_mass = mass.sum()
    com_vel = (vel * mass_2d).sum(dim=1) / torch.clamp(total_mass, min=1e-6)
    return com_vel
