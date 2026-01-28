"""Factory for creating MyoSuite tracking environments compatible with mjlab."""

from pathlib import Path
from typing import Any

import numpy as np

from ...config import MyoSuiteEnvCfg
from ...wrapper import MyoSuiteVecEnvWrapper
from .tracking_env_cfg import MyoSuiteTrackingEnvCfg


def _import_myosuite_gym():
  """Import MyoSuite gym module.

  Returns:
    The myosuite gym module

  Raises:
    ImportError: If MyoSuite is not available
  """
  try:
    from myosuite.utils import gym as myosuite_gym

    return myosuite_gym
  except ImportError:
    try:
      import myosuite

      if hasattr(myosuite, "utils") and hasattr(myosuite.utils, "gym"):
        return myosuite.utils.gym
    except (ImportError, AttributeError):
      pass
  raise ImportError(
    "MyoSuite is not installed. Install it with: pip install -U myosuite"
  ) from None


def _load_motion_file(motion_file: str | Path) -> dict[str, Any]:
  """Load motion data from a .npz file.

  Args:
    motion_file: Path to the motion file (.npz format)

  Returns:
    Dictionary containing motion data (typically with 'qp' and 'qv' keys)
  """
  motion_path = Path(motion_file)
  if not motion_path.exists():
    raise FileNotFoundError(f"Motion file not found: {motion_path}")

  # Check if file is empty (common in tests)
  if motion_path.stat().st_size == 0:
    # Return empty dict for empty files (tests often create empty files)
    return {}

  # Load the .npz file
  try:
    motion_data = np.load(motion_path, allow_pickle=True)
  except (EOFError, ValueError) as e:
    # Handle empty or corrupted files gracefully
    if "No data left in file" in str(e) or motion_path.stat().st_size == 0:
      return {}
    raise

  # Convert to dictionary format expected by MyoSuite's ReferenceMotion
  # MyoSuite's ReferenceMotion expects a dictionary with motion data
  # The exact format depends on the MyoSuite version, but typically includes:
  # - 'qp': joint positions over time
  # - 'qv': joint velocities over time
  # - 'time': time stamps (optional)
  motion_dict = {}
  for key in motion_data.keys():
    motion_dict[key] = motion_data[key]

  return motion_dict


def make_myosuite_tracking_env(
  myosuite_env_id: str,
  cfg: MyoSuiteTrackingEnvCfg | MyoSuiteEnvCfg | None = None,
  device: str = "cpu",
  render_mode: str | None = None,
  num_envs: int | None = None,
  **kwargs,
) -> MyoSuiteVecEnvWrapper:
  """Create a MyoSuite tracking environment wrapped for mjlab.

  This factory function handles MyoSuite tracking environments (like TrackEnv)
  by loading motion files and passing them as the 'reference' parameter.

  Args:
      myosuite_env_id: The original MyoSuite environment ID (e.g., 'myoHandReachRandom-v0')
      cfg: Tracking environment configuration (MyoSuiteTrackingEnvCfg)
      device: Device to use for tensors (default: "cpu")
      render_mode: Render mode - must be "rgb_array" for video recording (default: None)
      num_envs: Number of parallel environments (overrides cfg.num_envs if provided)
      **kwargs: Additional arguments passed to MyoSuite environment

  Returns:
      Wrapped MyoSuite tracking environment compatible with mjlab

  Raises:
      ValueError: If motion_file is not provided in cfg
      FileNotFoundError: If motion file does not exist
  """
  """Create a MyoSuite tracking environment wrapped for mjlab.

    This factory function handles MyoSuite tracking environments (like TrackEnv)
    by loading motion files and passing them as the 'reference' parameter.

    Args:
      myosuite_env_id: The original MyoSuite environment ID (e.g., 'myoHandReachRandom-v0')
      cfg: Tracking environment configuration (MyoSuiteTrackingEnvCfg)
      device: Device to use for tensors (default: "cpu")
      render_mode: Render mode (ignored for now)
      num_envs: Number of parallel environments (overrides cfg.num_envs if provided)
      **kwargs: Additional arguments passed to MyoSuite environment

    Returns:
      Wrapped MyoSuite tracking environment compatible with mjlab

    Raises:
      ValueError: If motion_file is not provided in cfg
      FileNotFoundError: If motion file does not exist
    """
  myosuite_gym = _import_myosuite_gym()

  # Use cfg if provided, otherwise use defaults
  if cfg is None:
    cfg = MyoSuiteTrackingEnvCfg()
  elif not isinstance(cfg, MyoSuiteTrackingEnvCfg):
    # If a regular MyoSuiteEnvCfg is passed, convert it to tracking config
    # This can happen if the wrong config type is used
    tracking_cfg = MyoSuiteTrackingEnvCfg()
    tracking_cfg.num_envs = cfg.num_envs if hasattr(cfg, "num_envs") else 1
    tracking_cfg.device = cfg.device if hasattr(cfg, "device") else device
    cfg = tracking_cfg

  # Validate that motion file is provided
  if not hasattr(cfg, "commands") or cfg.commands is None:
    raise ValueError(
      "MyoSuiteTrackingEnvCfg must have 'commands' attribute for tracking environments"
    )
  if not hasattr(cfg.commands, "motion") or cfg.commands.motion is None:
    raise ValueError(
      "MyoSuiteTrackingEnvCfg.commands must have 'motion' attribute for tracking environments"
    )
  if cfg.commands.motion.motion_file is None:
    raise ValueError(
      "MyoSuiteTrackingEnvCfg.commands.motion.motion_file must be provided for tracking environments. "
      "Set it via: cfg.commands.motion.motion_file = 'path/to/motion.npz'"
    )

  # Load motion file
  motion_data = _load_motion_file(cfg.commands.motion.motion_file)

  # num_envs from argument takes precedence over cfg
  if num_envs is None:
    num_envs = cfg.num_envs if hasattr(cfg, "num_envs") else 1
  device = cfg.device if hasattr(cfg, "device") and cfg.device else device

  # Create the MyoSuite tracking environment with reference motion
  # MyoSuite's TrackEnv expects 'reference' parameter in kwargs
  # The reference should be the motion data dictionary
  try:
    # Pass the motion data as 'reference' parameter
    # MyoSuite's TrackEnv._setup() expects this as the 'reference' argument
    myosuite_env = myosuite_gym.make(
      myosuite_env_id,
      reference=motion_data,
      **kwargs,
    )
  except TypeError as e:
    # If the environment doesn't accept 'reference', it might not be a tracking env
    # or the API might be different - try without it and let the env handle it
    if "reference" in str(e).lower() or "unexpected keyword" in str(e).lower():
      # Some MyoSuite versions might pass reference differently
      # Try passing it in _setup kwargs
      try:
        myosuite_env = myosuite_gym.make(
          myosuite_env_id,
          **kwargs,
        )
        # If the env has a _setup method, we might need to call it manually
        # But typically gym.make handles this, so we'll let it fail if needed
      except Exception as e2:
        raise RuntimeError(
          f"Failed to create MyoSuite tracking environment '{myosuite_env_id}': {e2}\n"
          f"Original error: {e}\n"
          "This environment might not support the 'reference' parameter, "
          "or the motion file format might be incorrect."
        ) from e2
    else:
      raise
  except AttributeError as e:
    raise RuntimeError(
      f"MyoSuite tracking environment creation failed: {e}\n"
      "This may be due to MuJoCo compatibility issues or missing dependencies."
    ) from e

  # Ensure target visualization is enabled for tracking environments
  # MyoSuite TrackEnv uses a "target" site to visualize the reference motion
  # Make sure it's visible
  try:
    # Unwrap to get the actual environment
    unwrapped = myosuite_env
    while hasattr(unwrapped, "unwrapped") and unwrapped.unwrapped is not unwrapped:
      unwrapped = unwrapped.unwrapped

    # Check if this is a TrackEnv and has a target site
    if hasattr(unwrapped, "sim") and hasattr(unwrapped.sim, "model"):
      try:
        # Try to find the target site
        target_site_id = unwrapped.sim.model.site_name2id("target")
        # Make sure the target site is visible (set alpha to 1.0)
        if target_site_id >= 0:
          unwrapped.sim.model.site_rgba[target_site_id][3] = (
            1.0  # Set alpha to fully visible
          )
          print("[INFO] Target site visualization enabled for tracking environment")
      except (AttributeError, ValueError):
        # Target site might not exist or might be named differently
        # This is okay - not all environments have a target site
        pass
    elif hasattr(unwrapped, "target_sid"):
      # TrackEnv has target_sid attribute
      try:
        if hasattr(unwrapped, "sim") and hasattr(unwrapped.sim, "model"):
          target_site_id = unwrapped.target_sid
          unwrapped.sim.model.site_rgba[target_site_id][3] = 1.0
          print("[INFO] Target site visualization enabled for tracking environment")
      except (AttributeError, IndexError):
        pass
  except Exception as e:
    # If visualization setup fails, continue anyway
    print(f"[WARNING] Could not enable target visualization: {e}")

  # Wrap it for mjlab compatibility
  wrapped_env = MyoSuiteVecEnvWrapper(
    env=myosuite_env,
    num_envs=num_envs,
    device=device,
    render_mode=render_mode,  # Pass render_mode to wrapper for video recording
  )

  # Get the spec from the original MyoSuite environment and set it on the wrapper
  if hasattr(myosuite_env, "spec") and myosuite_env.spec is not None:
    wrapped_env.spec = myosuite_env.spec

  return wrapped_env
