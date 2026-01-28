"""Auto-registration utilities for MyoSuite environments."""

import gymnasium as gym


def _is_tracking_env(env_id: str) -> bool:
  """Check if a MyoSuite environment ID corresponds to a tracking environment.

  MyoSuite tracking environments (TrackEnv) are those that accept a 'reference'
  parameter. Since we can't easily detect this at registration time, we use
  a conservative approach: only environments with 'myodm' (only tracking environment)in their name are
  considered tracking environments by default.

  However, any MyoSuite environment can be used as a tracking environment
  if a MyoSuiteTrackingEnvCfg with a motion file is provided when creating it.

  Args:
    env_id: The MyoSuite environment ID

  Returns:
    True if this is likely a tracking environment, False otherwise
  """
  # TODO: Add a check to see if the environment is a tracking environment by checking the entry point
  # MyoSuite tracking environments (TrackEnv) are in the myodm module
  # Look for 'myodm' in the environment ID as a strong indicator
  env_id_lower = env_id.lower()
  if "myodm" in env_id_lower:
    return True

  # We could also check the entry point, but that's expensive and may not work
  # for all MyoSuite versions. The myodm check is sufficient for now.
  return False


def _try_import_myosuite():
  """Try to import MyoSuite, supporting both standard and mjx/warp versions.

  Returns:
    bool: True if MyoSuite is available, False otherwise
  """
  # Try standard import
  try:
    import myosuite  # noqa: F401

    return True
  except ImportError:
    pass

  # Try mjx/warp compatible import
  try:
    from myosuite.utils import gym as myosuite_gym  # noqa: F401

    return True
  except ImportError:
    pass

  # Try alternative import paths
  try:
    import myosuite

    if hasattr(myosuite, "utils"):
      return True
  except (ImportError, AttributeError):
    pass

  return False


def register_myosuite_envs(prefix: str = "Mjlab-MyoSuite"):
  """Automatically discover and register all MyoSuite environments.

  Supports both standard MyoSuite and mjx/warp compatible versions.

  Args:
    prefix: Prefix to use for registered environment IDs
  """
  # Try to import myosuite to ensure environments are registered
  # MyoSuite registers environments when imported, but the import path may vary
  myosuite_available = _try_import_myosuite()

  # Also try to trigger registration by importing the gym module
  if myosuite_available:
    try:
      from myosuite.utils import gym as _  # noqa: F401
    except ImportError:
      try:
        import myosuite

        if hasattr(myosuite, "utils") and hasattr(myosuite.utils, "gym"):
          _ = myosuite.utils.gym  # noqa: F401
      except (ImportError, AttributeError):
        pass

  # Get all MyoSuite environments from their registry
  # Even if import fails, environments might already be registered
  myosuite_envs = []
  for env_id in gym.registry.keys():
    if "myo" in env_id.lower() and env_id not in myosuite_envs:
      # Skip our own registered environments
      if not env_id.startswith(prefix):
        myosuite_envs.append(env_id)

  # If no MyoSuite environments found and import failed, skip registration
  if not myosuite_envs and not myosuite_available:
    print("[INFO] MyoSuite not installed. Skipping MyoSuite environment registration.")
    return

  # Register all MyoSuite environments as regular environments
  for env_id in myosuite_envs:
    # Create a new ID with mjlab prefix
    mjlab_env_id = (
      f"{prefix}-{env_id.split('/')[-1]}" if "/" in env_id else f"{prefix}-{env_id}"
    )

    # Skip if already registered
    if mjlab_env_id in gym.registry:
      continue

    # Register with custom entry point that uses our wrapper
    gym.register(
      id=mjlab_env_id,
      entry_point="mjlab_myosuite.env_factory:make_myosuite_env",
      disable_env_checker=True,
      kwargs={
        "myosuite_env_id": env_id,
        "env_cfg_entry_point": "mjlab_myosuite.config:MyoSuiteEnvCfg",
        "rl_cfg_entry_point": "mjlab_myosuite.config:get_default_myosuite_rl_cfg",
      },
    )

  # Register ALL environments as tracking environments too (with Tracking prefix)
  # This allows any MyoSuite environment to be used for tracking when a motion file is provided
  tracking_prefix = f"{prefix}-Tracking"
  for env_id in myosuite_envs:
    # Create a new ID with Tracking prefix
    mjlab_tracking_env_id = (
      f"{tracking_prefix}-{env_id.split('/')[-1]}"
      if "/" in env_id
      else f"{tracking_prefix}-{env_id}"
    )

    # Skip if already registered
    if mjlab_tracking_env_id in gym.registry:
      continue

    # Register with tracking entry point that uses our tracking wrapper
    gym.register(
      id=mjlab_tracking_env_id,
      entry_point="mjlab_myosuite.tasks.tracking.env_factory:make_myosuite_tracking_env",
      disable_env_checker=True,
      kwargs={
        "myosuite_env_id": env_id,
        "env_cfg_entry_point": "mjlab_myosuite.tasks.tracking.tracking_env_cfg:MyoSuiteTrackingEnvCfg",
        "rl_cfg_entry_point": "mjlab_myosuite.config:get_default_myosuite_rl_cfg",
      },
    )

  print(
    f"[INFO] Registered {len(myosuite_envs)} regular MyoSuite environments with prefix '{prefix}'"
  )
  print(
    f"[INFO] Registered {len(myosuite_envs)} tracking MyoSuite environments with prefix '{tracking_prefix}'"
  )
