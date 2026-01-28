"""ONNX export functionality for MyoSuite environments."""

import os
from typing import Any

# Try to import ONNX exporter from mjlab, with fallback
_onnx_export_available = False
try:
  from mjlab.utils.lab_api.rl.exporter import export_policy_as_onnx

  _onnx_export_available = True
except ImportError:
  # Fallback: try alternative import paths
  try:
    from mjlab.rl.exporter_utils import export_policy_as_onnx

    _onnx_export_available = True
  except ImportError:
    # If ONNX export is not available, create a stub function
    def export_policy_as_onnx(*args, **kwargs) -> None:  # type: ignore[assignment]
      raise ImportError(
        "ONNX export functionality is not available. "
        "Please ensure mjlab is properly installed with ONNX support."
      )


def export_myosuite_policy_as_onnx(
  actor_critic: object,
  path: str,
  normalizer: object | None = None,
  filename="policy.onnx",
  verbose=False,
):
  """Export MyoSuite policy to ONNX format.

  Args:
    actor_critic: The actor-critic policy module.
    normalizer: The empirical normalizer module. If None, Identity is used.
    path: The path to the saving directory.
    filename: The name of exported ONNX file. Defaults to "policy.onnx".
    verbose: Whether to print the model summary. Defaults to False.

  Raises:
    ImportError: If ONNX export functionality is not available.
  """
  if not _onnx_export_available:
    raise ImportError(
      "ONNX export functionality is not available. "
      "Please ensure mjlab is properly installed with ONNX support."
    )
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  export_policy_as_onnx(
    policy=actor_critic,
    normalizer=normalizer,
    path=path,
    filename=filename,
    verbose=verbose,
  )


def attach_myosuite_onnx_metadata(
  env: Any, run_path: str, path: str, filename="policy.onnx"
) -> None:
  """Attach MyoSuite-specific metadata to ONNX model.

  Args:
    env: The RL environment (MyoSuiteVecEnvWrapper).
    run_path: W&B run path or other identifier.
    path: Directory containing the ONNX file.
    filename: Name of the ONNX file.
  """
  try:
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

    onnx_path = os.path.join(path, filename)
    # Try to get base metadata if env has full ManagerBasedRlEnv structure
    # For MyoSuite, we use fallback metadata since we have mock managers
    try:
      # Check if env has the required structure for get_base_metadata
      if (
        hasattr(env, "scene")
        and hasattr(env, "action_manager")
        and hasattr(env, "sim")
        and hasattr(env.scene, "__getitem__")
      ):
        # Try to access scene["robot"] to verify it's a real ManagerBasedRlEnv
        try:
          robot = env.scene["robot"]
          # Check if it's a real robot (has spec with actuators) or mock
          if (
            hasattr(robot, "spec")
            and hasattr(robot.spec, "actuators")
            and len(robot.spec.actuators) > 0
          ):
            # Real ManagerBasedRlEnv, try to get base metadata
            try:
              metadata = get_base_metadata(env, run_path)
            except (AttributeError, KeyError, TypeError, AssertionError):
              # Fallback if get_base_metadata fails
              metadata = _get_myosuite_metadata(env, run_path)
          else:
            # Mock scene, use fallback
            metadata = _get_myosuite_metadata(env, run_path)
        except (KeyError, AttributeError, TypeError):
          # Mock scene or missing robot, use fallback
          metadata = _get_myosuite_metadata(env, run_path)
      else:
        # Missing required attributes, use fallback
        metadata = _get_myosuite_metadata(env, run_path)
    except Exception:
      # Any error, use fallback
      metadata = _get_myosuite_metadata(env, run_path)

    attach_metadata_to_onnx(onnx_path, metadata)
  except ImportError:
    # ONNX metadata utilities not available, skip metadata attachment
    pass


def _get_myosuite_metadata(env: Any, run_path: str) -> dict[str, Any]:
  """Get minimal metadata for MyoSuite environments.

  Args:
    env: The MyoSuite environment wrapper.
    run_path: W&B run path or other identifier.

  Returns:
    Dictionary of metadata fields.
  """
  # Get action space info
  action_space = getattr(env, "single_action_space", getattr(env, "action_space", None))
  action_dim = None
  if action_space is not None:
    if hasattr(action_space, "shape") and action_space.shape is not None:
      try:
        action_dim = action_space.shape[0] if len(action_space.shape) > 0 else None
      except (TypeError, AttributeError):
        action_dim = None
    elif hasattr(action_space, "n"):
      action_dim = action_space.n

  # Get observation space info
  obs_space = getattr(
    env, "single_observation_space", getattr(env, "observation_space", None)
  )
  obs_dim = None
  if obs_space is not None:
    if hasattr(obs_space, "shape") and obs_space.shape is not None:
      try:
        obs_dim = obs_space.shape[0] if len(obs_space.shape) > 0 else None
      except (TypeError, AttributeError):
        obs_dim = None
    elif isinstance(obs_space, dict) and hasattr(obs_space, "spaces"):
      # For dict spaces, get policy observation dimension
      if "policy" in obs_space.spaces:
        policy_obs_space = obs_space.spaces["policy"]
        if hasattr(policy_obs_space, "shape") and policy_obs_space.shape is not None:
          try:
            obs_dim = (
              policy_obs_space.shape[0] if len(policy_obs_space.shape) > 0 else None
            )
          except (TypeError, AttributeError):
            obs_dim = None

  # Get action bounds
  action_scale = None
  if (
    action_space is not None
    and hasattr(action_space, "low")
    and hasattr(action_space, "high")
  ):
    # Compute scale as (high - low) / 2
    if hasattr(action_space.low, "__len__") and hasattr(action_space.high, "__len__"):
      import numpy as np

      low = np.array(action_space.low)
      high = np.array(action_space.high)
      action_scale = ((high - low) / 2.0).tolist()

  metadata = {
    "run_path": run_path,
    "task_type": "myosuite",
    "num_envs": getattr(env, "num_envs", 1),
    "device": str(getattr(env, "device", "cpu")),
  }

  if action_dim is not None:
    metadata["action_dim"] = action_dim
  if obs_dim is not None:
    metadata["observation_dim"] = obs_dim
  if action_scale is not None:
    metadata["action_scale"] = action_scale

  # Try to get environment ID if available
  try:
    if hasattr(env, "spec") and env.spec is not None and hasattr(env.spec, "id"):
      metadata["env_id"] = env.spec.id
    elif (
      hasattr(env, "env")
      and hasattr(env.env, "spec")
      and env.env.spec is not None
      and hasattr(env.env.spec, "id")
    ):
      metadata["env_id"] = env.env.spec.id
  except (AttributeError, TypeError):
    pass  # Skip env_id if not available

  return metadata
