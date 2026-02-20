"""Utilities for discovering MyoSuite environments (no gym registration).

Environments are created only via make_myosuite_env(); MyoSuite's native
gym registration is used by MyoSuite itself.
"""


def _is_tracking_env(env_id: str) -> bool:
  """Check if a MyoSuite environment ID corresponds to a tracking environment.

  MyoSuite tracking environments (TrackEnv) are those that accept a 'reference'
  parameter. Since we can't easily detect this at registration time, we use
  a conservative approach: only environments with 'myodm' in their name are
  considered tracking environments by default.

  However, any MyoSuite environment can be used as a tracking environment
  if a MyoSuiteTrackingEnvCfg with a motion file is provided when creating it.

  Args:
    env_id: The MyoSuite environment ID

  Returns:
    True if this is likely a tracking environment, False otherwise
  """
  env_id_lower = env_id.lower()
  if "myodm" in env_id_lower:
    return True
  return False


def _try_import_myosuite() -> bool:
  """Try to import MyoSuite, supporting both standard and mjx/warp versions.

  Returns:
    True if MyoSuite is available, False otherwise.
  """
  try:
    import myosuite  # noqa: F401

    return True
  except ImportError:
    pass
  try:
    from myosuite.utils import gym as myosuite_gym  # noqa: F401

    return True
  except ImportError:
    pass
  try:
    import myosuite

    if hasattr(myosuite, "utils"):
      return True
  except (ImportError, AttributeError):
    pass
  return False


def get_myosuite_env_ids() -> list[str]:
  """Return list of MyoSuite environment IDs.

  MyoSuite registers its envs with gymnasium on import. We collect IDs
  that look like MyoSuite envs (contain 'myo'), excluding any we used to
  add (Mjlab-MyoSuite-*).

  Returns:
    List of environment ID strings (e.g. ['myoElbowPose1D6MRandom-v0', ...]).
    Empty if MyoSuite is not available.
  """
  if not _try_import_myosuite():
    return []
  try:
    from myosuite.utils import gym as _  # noqa: F401 - trigger registration
  except ImportError:
    pass
  try:
    import gymnasium as gym

    return [
      e
      for e in gym.registry.keys()
      if "myo" in e.lower() and not e.startswith("Mjlab-MyoSuite")
    ]
  except Exception:
    return []
