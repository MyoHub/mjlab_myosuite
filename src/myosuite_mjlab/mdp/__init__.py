"""Extra MDP building blocks for musculoskeletal tasks (standalone, no fork)."""

from myosuite_mjlab.mdp.observations import (
    actuator_force,
    com_lin_vel_w,
    com_pos_w,
    tendon_length,
    tendon_velocity,
)
from myosuite_mjlab.mdp.curriculums import (
    action_scale_curriculum,
    push_velocity_curriculum,
)
from myosuite_mjlab.mdp.actions import SynergyTendonEffortActionCfg
from myosuite_mjlab.mdp.rewards import cyclic_hip_flexion_penalty

__all__ = [
    "tendon_length",
    "tendon_velocity",
    "actuator_force",
    "com_pos_w",
    "com_lin_vel_w",
    "action_scale_curriculum",
    "push_velocity_curriculum",
    "SynergyTendonEffortActionCfg",
    "cyclic_hip_flexion_penalty",
]
