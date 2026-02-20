"""Factory for creating MyoSuite environments compatible with mjlab."""

import os
from typing import TYPE_CHECKING

# Set MUJOCO_GL=egl early for headless rendering support
# This must be done BEFORE any MuJoCo imports or operations
if "MUJOCO_GL" not in os.environ:
  os.environ["MUJOCO_GL"] = "egl"

from .config import MyoSuiteEnvCfg, PhysicsBackend
from .wrapper import MyoSuiteVecEnvWrapper

if TYPE_CHECKING:
  pass


def detect_physics_backend() -> PhysicsBackend:
  """Detect whether MyoSuite provides a WARP/mjx backend (e.g. from mjx branch).

  Returns:
    PhysicsBackend.WARP if warp/mjx is available, else PhysicsBackend.CPU.
  See: https://github.com/MyoHub/myosuite/tree/mjx
  """
  try:
    import myosuite  # noqa: F401
  except ImportError:
    return PhysicsBackend.CPU
  # mjx branch may expose warp or mjx-specific attributes
  if getattr(myosuite, "HAS_WARP", False) or getattr(myosuite, "HAS_MJX", False):
    return PhysicsBackend.WARP
  try:
    import warp as wp  # noqa: F401

    # If we have warp and myosuite, mjx branch might be in use; be conservative
    return PhysicsBackend.CPU
  except ImportError:
    pass
  return PhysicsBackend.CPU


def _import_myosuite_gym(prefer_warp: bool = False):
  """Import MyoSuite gym module, trying mjx/warp versions first when prefer_warp, then standard.

  Args:
    prefer_warp: If True, prefer MyoSuite from mjx branch (warp-backed) when available.

  Returns:
    The myosuite gym module

  Raises:
    ImportError: If no MyoSuite version is available
  """
  # Try mjx/warp compatible version first (from mjx branch: https://github.com/MyoHub/myosuite/tree/mjx)
  try:
    from myosuite.utils import gym as myosuite_gym

    return myosuite_gym
  except ImportError:
    pass

  try:
    import myosuite

    if hasattr(myosuite, "utils") and hasattr(myosuite.utils, "gym"):
      return myosuite.utils.gym
  except (ImportError, AttributeError):
    pass

  try:
    from myosuite import utils

    if hasattr(utils, "gym"):
      return utils.gym
  except (ImportError, AttributeError):
    pass

  raise ImportError(
    "MyoSuite is not installed. Install it with: pip install -U myosuite\n"
    "For mjx/warp compatible versions, use the mjx branch:\n"
    "  https://github.com/MyoHub/myosuite/tree/mjx\n"
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

  # num_envs and device from arguments take precedence over cfg
  if num_envs is None:
    num_envs = cfg.num_envs if hasattr(cfg, "num_envs") else 1
  # Use device from cfg when caller did not pass device (default "cpu")
  if device == "cpu" and cfg is not None:
    cfg_device = getattr(cfg, "device", "cpu")
    if cfg_device is not None and str(cfg_device).lower().startswith("cuda"):
      device = str(cfg_device)
  # Sync device to cfg for consistency
  object.__setattr__(cfg, "device", device)

  # Physics backend: kwargs override, then config, then auto-detect
  physics_backend = kwargs.pop("physics_backend", None)
  if physics_backend is None:
    physics_backend = getattr(cfg, "physics_backend", None)
  if physics_backend is None:
    physics_backend = detect_physics_backend()

  # Try to create the environment with compatibility workarounds
  try:
    # Create the base MyoSuite environment (same gym.make for CPU and mjx; mjx branch uses warp under the hood)
    myosuite_env = myosuite_gym.make(myosuite_env_id, **kwargs)
  except AttributeError as e:
    print(e)
    raise RuntimeError(
      f"MyoSuite environment creation failed: {e}\n"
      "This may be due to MuJoCo compatibility issues. "
      "See docs/myosuite_troubleshooting.md for solutions."
    ) from e

  # Wrap it for mjlab compatibility (backend used for data path: shuttle vs bridge when supported)
  wrapped_env = MyoSuiteVecEnvWrapper(
    env=myosuite_env,
    num_envs=num_envs,
    device=device,
    render_mode=render_mode,
    physics_backend=physics_backend,
  )

  # Get the spec from the original MyoSuite environment and set it on the wrapper
  # This is required by gymnasium's environment checker
  if hasattr(myosuite_env, "spec") and myosuite_env.spec is not None:
    wrapped_env.spec = myosuite_env.spec

  return wrapped_env


def task_id_to_myosuite_id(task_id: str) -> tuple[str, bool]:
  """Parse mjlab MyoSuite task_id into base env id and tracking flag.

  Args:
    task_id: e.g. "Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0" or
      "Mjlab-MyoSuite-Tracking-myoElbowPose1D6MRandom-v0"

  Returns:
    (myosuite_env_id, is_tracking), e.g. ("myoElbowPose1D6MRandom-v0", False).
  """
  if not task_id.startswith("Mjlab-MyoSuite"):
    return task_id, False
  rest = task_id[len("Mjlab-MyoSuite-") :]
  if rest.startswith("Tracking-"):
    return rest[len("Tracking-") :], True
  return rest, False


def make_myosuite_env_from_task_id(
  task_id: str,
  cfg=None,
  device: str = "cpu",
  render_mode: str | None = None,
  num_envs: int | None = None,
  **kwargs,
) -> MyoSuiteVecEnvWrapper:
  """Create a wrapped MyoSuite env from mjlab task_id (e.g. Mjlab-MyoSuite-...).

  Use this when you have a task_id from the CLI or mjlab registry; for direct
  creation prefer make_myosuite_env(myosuite_env_id, ...).
  """
  myosuite_env_id, is_tracking = task_id_to_myosuite_id(task_id)
  if is_tracking:
    from .tasks.tracking.env_factory import make_myosuite_tracking_env

    return make_myosuite_tracking_env(
      myosuite_env_id=myosuite_env_id,
      cfg=cfg,
      device=device,
      render_mode=render_mode,
      num_envs=num_envs,
      **kwargs,
    )
  return make_myosuite_env(
    myosuite_env_id=myosuite_env_id,
    cfg=cfg,
    device=device,
    render_mode=render_mode,
    num_envs=num_envs,
    **kwargs,
  )
