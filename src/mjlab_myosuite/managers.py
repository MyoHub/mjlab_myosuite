"""Custom managers for MyoSuite integration with ManagerBasedRlEnv."""

from typing import Any

import torch


def _get_observation_term_base():
  """Get ObservationTermCfg base class."""
  try:
    from mjlab.managers.observation_term import ObservationTermCfg

    return ObservationTermCfg
  except ImportError:
    return None


def _get_action_term_base():
  """Get ActionTermCfg base class."""
  try:
    # Try different import paths
    try:
      from mjlab.managers.action_term import ActionTermCfg

      return ActionTermCfg
    except ImportError:
      try:
        from mjlab.managers import ActionTermCfg

        return ActionTermCfg
      except ImportError:
        # Try to get from managers module
        import mjlab.managers as managers_module

        if hasattr(managers_module, "ActionTermCfg"):
          return managers_module.ActionTermCfg
        return None
  except ImportError:
    return None


# Create observation term for MyoSuite
_ObservationTermCfg = _get_observation_term_base()

if _ObservationTermCfg is not None:

  class MyoSuiteObservationTermCfg(_ObservationTermCfg):
    """Observation term that extracts observations from MyoSuite environment.

    This term extracts observations from the wrapped MyoSuite environment
    and makes them available to ManagerBasedRlEnv's observation manager.
    """

    def compute(self, env: Any) -> torch.Tensor:
      """Extract observations from MyoSuite environment.

      Args:
          env: ManagerBasedRlEnv instance that has myosuite_env attribute

      Returns:
          Observation tensor from MyoSuite environment
      """
      if not hasattr(env, "myosuite_env"):
        raise RuntimeError(
          "MyoSuite environment not found. "
          "ManagerBasedRlEnv must have myosuite_env attribute."
        )

      # Get observations from MyoSuite environment
      myosuite_env = env.myosuite_env
      if hasattr(myosuite_env, "get_observations"):
        obs = myosuite_env.get_observations()
        # Return policy observations
        if isinstance(obs, dict):
          result = obs.get("policy", obs.get(list(obs.keys())[0]))
          if result is None:
            raise RuntimeError("No observations found in MyoSuite environment")
          return result
        if obs is None:
          raise RuntimeError("MyoSuite environment returned None observations")
        return obs
      else:
        # Fallback: get observations from step/reset
        # This shouldn't happen if MyoSuiteVecEnvWrapper is used
        raise RuntimeError("MyoSuite environment does not have get_observations method")

else:
  # Fallback if ObservationTermCfg is not available
  MyoSuiteObservationTermCfg = None  # type: ignore


# Create action term for MyoSuite
_ActionTermCfg = _get_action_term_base()

if _ActionTermCfg is not None:

  class MyoSuiteActionTermCfg(_ActionTermCfg):
    """Action term that passes actions to MyoSuite environment.

    This term processes actions and passes them through to the
    wrapped MyoSuite environment.
    """

    def process(self, env: Any, actions: torch.Tensor) -> torch.Tensor:
      """Process actions for MyoSuite environment.

      Args:
          env: ManagerBasedRlEnv instance
          actions: Action tensor from policy

      Returns:
          Processed action tensor (pass-through for MyoSuite)
      """
      # Actions are already in the right format for MyoSuite
      # Just return them as-is
      return actions

else:
  # Fallback if ActionTermCfg is not available
  MyoSuiteActionTermCfg = None  # type: ignore
