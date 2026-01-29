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

    # Get all MyoSuite environments from gymnasium registry
    # First, ensure they're registered in gymnasium registry
    from ..registration import register_myosuite_envs

    register_myosuite_envs()  # This registers in gymnasium registry

    # Now get the list of MyoSuite environments
    myosuite_envs = []
    for env_id in gym.registry.keys():
      if "myo" in env_id.lower() and not env_id.startswith("Mjlab-MyoSuite"):
        myosuite_envs.append(env_id)

    if not myosuite_envs:
      # No MyoSuite environments found, fall back to gymnasium registry
      print("[INFO] No MyoSuite environments found, using gymnasium registry")
      return

    # Register each MyoSuite environment with mjlab's native system
    registered_count = 0
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
        from mjlab.tasks.registry import load_env_cfg

        try:
          load_env_cfg(task_id)
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
      # Type ignore: MyoSuite configs are patched to work with ManagerBasedRlEnv
      register_mjlab_task(
        task_id=task_id,
        env_cfg=make_env_cfg(play=False),  # type: ignore[arg-type]
        play_env_cfg=make_env_cfg(play=True),  # type: ignore[arg-type]
        rl_cfg=make_rl_cfg(),
        runner_cls=VelocityOnPolicyRunner,
      )
      registered_count += 1

    print(
      f"[INFO] Registered {registered_count} MyoSuite tasks with mjlab's native system"
    )

    # Also register tracking tasks
    try:
      from ..tasks.tracking.rl import MyoSuiteMotionTrackingOnPolicyRunner
      from ..tasks.tracking.tracking_env_cfg import MyoSuiteTrackingEnvCfg

      # Get tracking tasks from gymnasium registry
      tracking_envs = []
      for env_id in gym.registry.keys():
        if env_id.startswith("Mjlab-MyoSuite-Tracking-"):
          # Extract the base MyoSuite env ID
          base_env_id = env_id.replace("Mjlab-MyoSuite-Tracking-", "")
          tracking_envs.append((env_id, base_env_id))

      if tracking_envs:
        tracking_registered_count = 0
        for tracking_env_id, base_env_id in tracking_envs:
          # Skip if already registered
          try:
            from mjlab.tasks.registry import load_env_cfg

            try:
              load_env_cfg(tracking_env_id)
              continue  # Already registered
            except (KeyError, ValueError, AttributeError):
              pass  # Not registered yet, continue
          except ImportError:
            pass  # Can't check, continue anyway

          # Create tracking environment config factory
          def make_tracking_env_cfg(base_env_id=base_env_id, play=False):
            """Create tracking environment config for this MyoSuite task."""
            cfg = MyoSuiteTrackingEnvCfg()
            return cfg

          # Create RL config factory (same as regular tasks)
          def make_tracking_rl_cfg():
            """Create RL config for this MyoSuite tracking task."""
            return get_default_myosuite_rl_cfg()

          # Register with mjlab's native system using tracking runner
          # Type ignore: MyoSuite configs are patched to work with ManagerBasedRlEnv
          register_mjlab_task(
            task_id=tracking_env_id,
            env_cfg=make_tracking_env_cfg(play=False),  # type: ignore[arg-type]
            play_env_cfg=make_tracking_env_cfg(play=True),  # type: ignore[arg-type]
            rl_cfg=make_tracking_rl_cfg(),
            runner_cls=MyoSuiteMotionTrackingOnPolicyRunner,
          )
          tracking_registered_count += 1

        print(
          f"[INFO] Registered {tracking_registered_count} MyoSuite tracking tasks with mjlab's native system"
        )
    except ImportError:
      # Tracking support not available, skip
      pass
    except Exception as e:
      # Log but don't fail
      import warnings

      warnings.warn(
        f"Failed to register tracking tasks: {e}", UserWarning, stacklevel=2
      )

  except ImportError:
    # mjlab's native registration not available, fall back to gymnasium registry
    from ..registration import register_myosuite_envs

    register_myosuite_envs()
    print("[INFO] Using gymnasium registry (mjlab native registration not available)")
    return

  except Exception as e:
    # Any error during registration, fall back to gymnasium registry
    from ..registration import register_myosuite_envs

    register_myosuite_envs()
    print(f"[INFO] Using gymnasium registry (mjlab native registration failed: {e})")
