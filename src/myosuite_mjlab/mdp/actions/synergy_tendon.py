"""Synergy-based tendon effort actions for MyoLegsTorso (standalone, no fork).

Low-dimensional action is expanded to full tendon effort via a fixed
linear mapping (290 tendons, 104 synergy groups).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_SYNERGY_TO_TENDON_INDICES: Dict[int, List[int]] = {
    0: list(range(0, 11)),
    1: list(range(11, 22)),
    2: [22],
    3: [23],
    4: [24, 25, 26, 27],
    5: [28, 29, 30, 31],
    6: [32, 33, 34, 35, 36, 37, 38, 39],
    7: [40, 41, 42, 43, 44, 45, 46, 47],
    8: list(range(48, 69)),
    9: list(range(69, 90)),
    10: [90, 91, 92, 93, 94],
    11: [95, 96, 97, 98, 99],
    12: [100, 101, 102, 103, 104, 105, 106],
    13: [107, 108, 109, 110, 111, 112, 113],
    14: [114, 115, 116, 117, 118],
    15: [119, 120, 121, 122, 123],
    16: [124, 125, 126, 127, 128, 129],
    17: [130, 131, 132, 133, 134, 135],
    18: list(range(136, 161)),
    19: list(range(161, 186)),
    20: [186, 187, 188, 189, 190, 191],
    21: [192, 193, 194, 195, 196, 197],
    22: [198, 199, 200, 201, 202, 203],
    23: [204, 205, 206, 207, 208, 209],
}
for _i in range(24, 104):
    _SYNERGY_TO_TENDON_INDICES[_i] = [210 + (_i - 24)]


@dataclass(kw_only=True)
class SynergyTendonEffortActionCfg(ActionTermCfg):
    """Configuration for synergy-based tendon effort control."""

    scale: float = 1.0

    def build(self, env: ManagerBasedRlEnv) -> SynergyTendonEffortAction:
        return SynergyTendonEffortAction(self, env)


class SynergyTendonEffortAction(ActionTerm):
    """Expand synergy actions to full tendon efforts and apply to the entity."""

    cfg: SynergyTendonEffortActionCfg

    def __init__(
        self,
        cfg: SynergyTendonEffortActionCfg,
        env: ManagerBasedRlEnv,
    ):
        super().__init__(cfg=cfg, env=env)
        mj_model = env.sim.mj_model
        self._tendon_ids = torch.arange(
            mj_model.ntendon, device=env.device, dtype=torch.long
        )
        self._num_tendons = int(mj_model.ntendon)
        if self._num_tendons != 290:
            raise ValueError(
                f"SynergyTendonEffortAction expects 290 tendons, got {self._num_tendons}."
            )
        self._synergy_to_indices: Dict[int, List[int]] = _SYNERGY_TO_TENDON_INDICES
        self._num_synergies = len(self._synergy_to_indices)
        covered = [
            idx for indices in self._synergy_to_indices.values() for idx in indices
        ]
        if sorted(covered) != list(range(self._num_tendons)):
            raise ValueError("Synergy mapping does not cover all tendons.")
        self._raw_actions = torch.zeros(
            env.num_envs,
            self._num_synergies,
            device=env.device,
            dtype=torch.float32,
        )
        self._expanded_actions = torch.zeros(
            env.num_envs,
            self._num_tendons,
            device=env.device,
            dtype=torch.float32,
        )

    @property
    def action_dim(self) -> int:
        return self._num_synergies

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape[1] != self._num_synergies:
            raise ValueError(
                f"Expected action dim {self._num_synergies}, got {actions.shape[1]}."
            )
        scale = getattr(self.cfg, "scale", 1.0)
        self._raw_actions[:] = actions * scale
        self._expanded_actions.zero_()
        for syn_idx, tendon_indices in self._synergy_to_indices.items():
            idxs = torch.as_tensor(
                tendon_indices,
                device=self._expanded_actions.device,
                dtype=torch.long,
            )
            self._expanded_actions[:, idxs] = self._raw_actions[
                :, syn_idx : syn_idx + 1
            ]

    def apply_actions(self) -> None:
        self._entity.set_tendon_effort_target(
            self._expanded_actions, tendon_ids=self._tendon_ids
        )
