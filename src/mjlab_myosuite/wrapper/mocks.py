"""Mock manager and scene classes for ManagerBasedRlEnv compatibility.

These are used by MyoSuiteVecEnvWrapper to provide the minimal interface that
mjlab's ManagerBasedRlEnv and related utilities expect (action_manager,
observation_manager, command_manager, scene).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch


class _MockActionManager:
  """Mock action manager for ManagerBasedRlEnv compatibility."""

  def __init__(self, action_space: gym.Space, num_envs: int):
    self.action_space = action_space
    self.num_envs = num_envs
    self.active_terms = ["joint_pos"]  # Default action term name

  def get_term(self, name: str) -> Any:  # noqa: ARG002 - interface compatibility
    """Get action term by name."""

    # Return a mock action term object
    class _MockActionTerm:
      def __init__(self, action_space: gym.Space):
        self._scale = self._get_action_scale(action_space)

      def _get_action_scale(self, action_space: gym.Space) -> torch.Tensor:
        """Get action scale from action space."""
        if hasattr(action_space, "low") and hasattr(action_space, "high"):
          low = np.array(action_space.low)
          high = np.array(action_space.high)
          # Return scale as (high - low) / 2
          scale = (high - low) / 2.0
          return torch.tensor(scale, dtype=torch.float32)
        return torch.ones(self._get_action_dim(action_space), dtype=torch.float32)

      def _get_action_dim(self, action_space: gym.Space) -> int:
        """Get action dimension."""
        if hasattr(action_space, "shape"):
          return int(np.prod(action_space.shape))  # type: ignore[arg-type]
        return 1

    return _MockActionTerm(self.action_space)


class _MockObservationManager:
  """Mock observation manager for ManagerBasedRlEnv compatibility."""

  def __init__(self, observation_space: gym.Space):
    self.observation_space = observation_space
    self.active_terms = {"policy": self._get_observation_names(observation_space)}

  def _get_observation_names(self, observation_space: gym.Space) -> list[str]:
    """Get observation names from observation space."""
    if isinstance(observation_space, gym.spaces.Dict):
      if "policy" in observation_space.spaces:
        policy_space = observation_space.spaces["policy"]
        if hasattr(policy_space, "shape") and policy_space.shape is not None:
          dim = int(np.prod(policy_space.shape))  # type: ignore[arg-type]
          return [f"obs_{i}" for i in range(dim)]
      # Fallback: use first space
      if observation_space.spaces:
        first_space = next(iter(observation_space.spaces.values()))
        if hasattr(first_space, "shape") and first_space.shape is not None:
          dim = int(np.prod(first_space.shape))  # type: ignore[arg-type]
          return [f"obs_{i}" for i in range(dim)]
    elif hasattr(observation_space, "shape") and observation_space.shape is not None:
      dim = int(np.prod(observation_space.shape))  # type: ignore[arg-type]
      return [f"obs_{i}" for i in range(dim)]
    return ["obs_0"]


class _MockCommandManager:
  """Mock command manager for ManagerBasedRlEnv compatibility."""

  def __init__(self) -> None:
    self.active_terms: list[str] = []  # MyoSuite doesn't use commands by default


class _MockScene:
  """Mock scene for ManagerBasedRlEnv compatibility."""

  def __init__(self, num_envs: int):
    self.num_envs = num_envs

  def __getitem__(self, key: str) -> Any:  # noqa: ARG002 - interface compatibility
    """Get entity by name (returns mock robot entity)."""

    # Return a mock robot entity
    class _MockRobot:
      def __init__(self) -> None:
        self.joint_names: list[str] = []  # Will be populated if needed
        self.spec = _MockSpec()
        self.data = _MockData()

    class _MockSpec:
      def __init__(self) -> None:
        self.actuators: list[Any] = []  # Empty actuators list

    class _MockData:
      def __init__(self) -> None:
        # Empty default positions
        self.default_joint_pos = torch.zeros(1, 0)  # (batch, dof)

    return _MockRobot()
