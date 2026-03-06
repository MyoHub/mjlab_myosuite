"""VecEnv wrapper that guarantees step() returns 5 values for rsl_rl>=4.

Public mjlab's RslRlVecEnvWrapper may return 4 values (obs, rew, dones, extras).
rsl_rl expects (obs, rewards, dones, infos, extras). This adapter inserts
empty infos when the underlying env returns 4 values.
"""

from __future__ import annotations

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from typing import Any, cast


class RslRlStepAdapter(VecEnv):
    """Wraps a VecEnv so step() always returns (obs, rewards, dones, infos, extras)."""

    def __init__(self, env: VecEnv) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.num_actions = env.num_actions

    @property
    def unwrapped(self) -> VecEnv:
        """Return the innermost env (ManagerBasedRlEnv) for runner state access."""
        out = self.env
        while hasattr(out, "unwrapped"):
            out = out.unwrapped  # type: ignore[union-attr]
        return out

    @property
    def observation_space(self):  # type: ignore[no-any-return]
        return self.env.observation_space

    @property
    def action_space(self):  # type: ignore[no-any-return]
        return self.env.action_space

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf  # type: ignore[union-attr]

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:  # type: ignore[override]
        self.env.episode_length_buf = value  # type: ignore[union-attr]

    def get_observations(self) -> TensorDict:
        return self.env.get_observations()  # type: ignore[union-attr]

    def reset(self) -> tuple[TensorDict, dict]:
        return self.env.reset()  # type: ignore[union-attr]

    def step(  # type: ignore[override]
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict, dict]:
        raw = self.env.step(actions)  # type: ignore[union-attr]
        result = cast("tuple[Any, ...]", raw)
        if len(result) == 4:
            obs, rew, dones, extras = result
            return obs, rew, dones, {}, extras
        obs, rew, dones, infos, extras = (
            result[0],
            result[1],
            result[2],
            result[3],
            result[4],
        )
        return obs, rew, dones, infos, extras

    def seed(self, seed: int = -1) -> int:
        return self.env.seed(seed)  # type: ignore[union-attr]

    def close(self) -> None:
        self.env.close()  # type: ignore[union-attr]
