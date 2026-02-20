"""Custom managers for MyoSuite integration with ManagerBasedRlEnv.

Aligns with mjlab's manager architecture (observation pipeline, reward, termination)
so that mjlab_myosuite works in the same way as mjlab. See docs/STRATEGY_MJLAB_ALIGNMENT.md.
"""

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


# -----------------------------------------------------------------------------
# Reward manager (mjlab-aligned: dt-scaling + NaN guard)
# -----------------------------------------------------------------------------


def compute_reward_mjlab_style(
  reward_raw: torch.Tensor,
  dt: float,
  scale_by_dt: bool = True,
  nan_guard: bool = True,
) -> torch.Tensor:
  """Apply mjlab-style reward processing: optional dt-scaling and NaN guard.

  Args:
    reward_raw: Raw reward from env (shape [num_envs] or [num_envs, 1]).
    dt: Environment step duration (for scale_by_dt).
    scale_by_dt: If True, multiply reward by dt (mjlab default).
    nan_guard: If True, replace NaN/Inf with 0.

  Returns:
    Processed reward tensor same shape as reward_raw.
  """
  out = reward_raw.float()
  if out.dim() == 2 and out.shape[-1] == 1:
    out = out.squeeze(-1)
  if scale_by_dt and dt > 0:
    out = out * dt
  if nan_guard:
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
  return out


# -----------------------------------------------------------------------------
# Termination manager (mjlab-aligned: reset_buf, terminated, time_outs)
# -----------------------------------------------------------------------------


def compute_termination_mjlab_style(
  num_envs: int,
  device: torch.device | str,
  terminated: torch.Tensor | None = None,
  truncated: torch.Tensor | None = None,
  time_outs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Build mjlab-style termination buffers from env done signals.

  mjlab uses reset_buf = terminated | truncated, and separates time_outs
  from other terminations. We follow the same contract.

  Args:
    num_envs: Number of environments.
    device: Torch device for buffers.
    terminated: [num_envs] bool (true if episode ended due to failure).
    truncated: [num_envs] bool (true if time limit or truncation).
    time_outs: [num_envs] bool (true if ended due to time limit only); if None, inferred from truncated.

  Returns:
    (reset_buf, terminated, time_outs) all [num_envs] bool on device.
  """
  if isinstance(device, str):
    device = torch.device(device)
  if terminated is None:
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
  else:
    terminated = terminated.to(device).bool().flatten()
  if truncated is None:
    truncated = torch.zeros(num_envs, dtype=torch.bool, device=device)
  else:
    truncated = truncated.to(device).bool().flatten()
  if time_outs is None:
    # By convention, treat truncated as time-out when we don't have separate signal
    time_outs = truncated.clone()
  else:
    time_outs = time_outs.to(device).bool().flatten()
  reset_buf = terminated | truncated
  return reset_buf, terminated, time_outs


# -----------------------------------------------------------------------------
# Observation pipeline (mjlab-aligned: compute → noise → clip → scale)
# Optional no-op stages for future use.
# -----------------------------------------------------------------------------


def apply_observation_pipeline(
  obs: torch.Tensor,
  noise_scale: float = 0.0,
  clip_range: tuple[float, float] | None = None,
  scale: float = 1.0,
) -> torch.Tensor:
  """Apply mjlab-style observation post-processing (noise, clip, scale).

  Defaults are no-ops (noise_scale=0, clip_range=None, scale=1.0) so that
  existing behavior is unchanged. The wrapper or config can enable these
  to match mjlab's observation manager pipeline.

  Args:
    obs: Observation tensor [num_envs, dim].
    noise_scale: Std of Gaussian noise to add (0 = no noise).
    clip_range: (low, high) to clip obs; None = no clip.
    scale: Multiplier for obs (1.0 = no scale).

  Returns:
    Processed observation tensor.
  """
  out = obs.float()
  if noise_scale > 0:
    out = out + noise_scale * torch.randn_like(out, device=out.device, dtype=out.dtype)
  if clip_range is not None:
    low, high = clip_range
    out = out.clamp(low, high)
  if scale != 1.0:
    out = out * scale
  return out
