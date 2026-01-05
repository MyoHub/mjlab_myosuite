"""Native mjlab task registration for MyoSuite environments."""

import gymnasium as gym

from ..config import MyoSuiteEnvCfg, get_default_myosuite_rl_cfg
from ..registration import _try_import_myosuite


def register_myosuite_tasks():
  """Register MyoSuite environments using mjlab's native task registration system.

  This function attempts to use mjlab's `register_mjlab_task` if available,
  otherwise falls back to gymnasium's registry (which still works with mjlab).
  """
  # Check if MyoSuite is available
  if not _try_import_myosuite():
    return

  # Try to use mjlab's native task registration
  try:
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
  except ImportError:
    # mjlab's native registration not available, fall back to gymnasium registry
    from ..registration import register_myosuite_envs

    register_myosuite_envs()
    print("[INFO] Using gymnasium registry (mjlab native registration not available)")
    return

    # Get all MyoSuite environments
    myosuite_envs = []
    for env_id in gym.registry.keys():
      if "myo" in env_id.lower() and not env_id.startswith("Mjlab-MyoSuite"):
        myosuite_envs.append(env_id)

    if not myosuite_envs:
      return

    # Register each MyoSuite environment with mjlab's native system
    for env_id in myosuite_envs:
      # Create task ID with mjlab prefix
      task_id = (
        f"Mjlab-MyoSuite-{env_id.split('/')[-1]}"
        if "/" in env_id
        else f"Mjlab-MyoSuite-{env_id}"
      )

      # Skip if already registered
      try:
        # Check if already registered by trying to load config
        from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
          load_cfg_from_registry,
        )

        try:
          load_cfg_from_registry(task_id, "env_cfg_entry_point")
          continue  # Already registered
        except (KeyError, ValueError, AttributeError):
          pass  # Not registered yet, continue
      except ImportError:
        pass  # Can't check, continue anyway

      # Create environment config factory
      def make_env_cfg(env_id=env_id, play=False):
        """Create environment config for this MyoSuite task."""
        cfg = MyoSuiteEnvCfg()
        # You can customize config per environment here if needed
        return cfg

      # Create RL config factory
      def make_rl_cfg():
        """Create RL config for this MyoSuite task."""
        return get_default_myosuite_rl_cfg()

      # Register with mjlab's native system
      register_mjlab_task(
        task_id=task_id,
        env_cfg=make_env_cfg(play=False),
        play_env_cfg=make_env_cfg(play=True),
        rl_cfg=make_rl_cfg(),
        runner_cls=VelocityOnPolicyRunner,
      )

    print(
      f"[INFO] Registered {len(myosuite_envs)} MyoSuite tasks with mjlab's native system"
    )

  except Exception as e:
    # Any error during registration, fall back to gymnasium registry
    from ..registration import register_myosuite_envs

    register_myosuite_envs()
    print(f"[INFO] Using gymnasium registry (mjlab native registration failed: {e})")
