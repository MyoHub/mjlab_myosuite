"""Factory for creating MyoSuite environments compatible with mjlab."""

import os
from typing import TYPE_CHECKING

# Set MUJOCO_GL=egl early for headless rendering support
# This must be done BEFORE any MuJoCo imports or operations
if "MUJOCO_GL" not in os.environ:
  os.environ["MUJOCO_GL"] = "egl"

from .config import MyoSuiteEnvCfg
from .wrapper import MyoSuiteVecEnvWrapper

if TYPE_CHECKING:
  pass


def _import_myosuite_gym():
  """Import MyoSuite gym module, trying mjx/warp versions first, then standard version.

  Returns:
    The myosuite gym module

  Raises:
    ImportError: If no MyoSuite version is available
  """
  # Try mjx/warp compatible version first (from mjx branch)
  try:
    from myosuite.utils import gym as myosuite_gym

    # Check if this is the mjx/warp version by looking for specific attributes
    # The mjx version might have different module structure
    return myosuite_gym
  except ImportError:
    pass

  # Try alternative import paths for mjx/warp versions
  try:
    # Some versions might have different import paths
    import myosuite

    if hasattr(myosuite, "utils") and hasattr(myosuite.utils, "gym"):
      return myosuite.utils.gym
  except (ImportError, AttributeError):
    pass

  # Final fallback - try direct import
  try:
    from myosuite import utils

    if hasattr(utils, "gym"):
      return utils.gym
  except (ImportError, AttributeError):
    pass

  raise ImportError(
    "MyoSuite is not installed. Install it with: pip install -U myosuite\n"
    "For mjx/warp compatible versions, use the mjx branch:\n"
    "  git clone https://github.com/MyoHub/myosuite.git\n"
    "  cd myosuite && git checkout mjx && pip install -e ."
  )


def make_myosuite_env(
  myosuite_env_id: str,
  cfg: MyoSuiteEnvCfg | None = None,
  device: str = "cpu",
  render_mode: str | None = None,
  num_envs: int | None = None,
  **kwargs,
) -> MyoSuiteVecEnvWrapper:
  """Create a MyoSuite environment wrapped for mjlab.

  Supports both standard MyoSuite and mjx/warp compatible versions.

  If a MyoSuiteTrackingEnvCfg with a motion file is provided, this will
  automatically delegate to the tracking environment factory.

  Args:
    myosuite_env_id: The original MyoSuite environment ID
    cfg: Environment configuration (MyoSuiteEnvCfg or MyoSuiteTrackingEnvCfg)
    device: Device to use for tensors (default: "cpu")
    render_mode: Render mode (ignored for now, MyoSuite handles rendering differently)
    num_envs: Number of parallel environments (overrides cfg.num_envs if provided)
    **kwargs: Additional arguments passed to MyoSuite environment

  Returns:
    Wrapped MyoSuite environment compatible with mjlab
  """
  # Check if this is a tracking config with a motion file
  # If so, delegate to the tracking factory
  if cfg is not None:
    # Try to import tracking config to check type
    try:
      from .tasks.tracking.tracking_env_cfg import MyoSuiteTrackingEnvCfg

      if isinstance(cfg, MyoSuiteTrackingEnvCfg):
        # Check if motion file is provided
        if (
          hasattr(cfg, "commands")
          and cfg.commands is not None
          and hasattr(cfg.commands, "motion")
          and cfg.commands.motion is not None
          and cfg.commands.motion.motion_file is not None
        ):
          # Delegate to tracking factory
          from .tasks.tracking.env_factory import make_myosuite_tracking_env

          return make_myosuite_tracking_env(
            myosuite_env_id=myosuite_env_id,
            cfg=cfg,
            device=device,
            render_mode=render_mode,
            num_envs=num_envs,
            **kwargs,
          )
    except ImportError:
      # Tracking module not available, continue with regular factory
      pass

  myosuite_gym = _import_myosuite_gym()

  # Don't set environment variables here - let the system use defaults
  # Setting DISPLAY=:99 or MUJOCO_GL=egl breaks the viewer when a real display is available
  # These should only be set when running in a truly headless environment (e.g., during training)
  # For play/viewer mode, the system should use the default display

  # Use cfg if provided, otherwise use defaults
  if cfg is None:
    cfg = MyoSuiteEnvCfg()

  # num_envs from argument takes precedence over cfg
  if num_envs is None:
    num_envs = cfg.num_envs if hasattr(cfg, "num_envs") else 1
  device = cfg.device if hasattr(cfg, "device") and cfg.device else device

  # Try to create the environment with compatibility workarounds
  try:
    # Create the base MyoSuite environment
    myosuite_env = myosuite_gym.make(myosuite_env_id, **kwargs)
  except AttributeError as e:
    print(e)
    raise RuntimeError(
      f"MyoSuite environment creation failed: {e}\n"
      "This may be due to MuJoCo compatibility issues. "
      "See docs/myosuite_troubleshooting.md for solutions."
    ) from e

  # Wrap it for mjlab compatibility
  wrapped_env = MyoSuiteVecEnvWrapper(
    env=myosuite_env,
    num_envs=num_envs,
    device=device,
    render_mode=render_mode,  # Pass render_mode to wrapper
  )

  # Get the spec from the original MyoSuite environment and set it on the wrapper
  # This is required by gymnasium's environment checker
  if hasattr(myosuite_env, "spec") and myosuite_env.spec is not None:
    wrapped_env.spec = myosuite_env.spec

  return wrapped_env
