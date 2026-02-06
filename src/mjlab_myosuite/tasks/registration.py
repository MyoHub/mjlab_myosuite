"""Native mjlab task registration for MyoSuite environments.

Registers MyoSuite tasks with mjlab's task registry (for CLI train/play).
Environments are created only via make_myosuite_env() / make_myosuite_env_from_task_id().
"""

from ..config import MyoSuiteEnvCfg, get_default_myosuite_rl_cfg
from ..registration import _try_import_myosuite, get_myosuite_env_ids


def register_myosuite_tasks() -> None:
  """Register MyoSuite environments using mjlab's native task registration.

  Discovers env IDs from MyoSuite (no gym registration). Task IDs use the
  form Mjlab-MyoSuite-<env_id> and Mjlab-MyoSuite-Tracking-<env_id>.
  """
  if not _try_import_myosuite():
    return

  try:
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
  except ImportError:
    return

  myosuite_envs = get_myosuite_env_ids()
  if not myosuite_envs:
    return

  registered_count = 0
  for env_id in myosuite_envs:
    task_id = (
      f"Mjlab-MyoSuite-{env_id.split('/')[-1]}"
      if "/" in env_id
      else f"Mjlab-MyoSuite-{env_id}"
    )
    try:
      from mjlab.tasks.registry import load_env_cfg

      try:
        load_env_cfg(task_id)
        continue
      except (KeyError, ValueError, AttributeError):
        pass
    except ImportError:
      pass

    def make_env_cfg(env_id=env_id, play=False):
      return MyoSuiteEnvCfg()

    def make_rl_cfg():
      return get_default_myosuite_rl_cfg()

    register_mjlab_task(
      task_id=task_id,
      env_cfg=make_env_cfg(play=False),  # type: ignore[arg-type]
      play_env_cfg=make_env_cfg(play=True),  # type: ignore[arg-type]
      rl_cfg=make_rl_cfg(),
      runner_cls=VelocityOnPolicyRunner,
    )
    registered_count += 1

  print(
    f"[INFO] Registered {registered_count} MyoSuite tasks with mjlab (task_id: Mjlab-MyoSuite-*)"
  )

  # Tracking tasks
  try:
    from mjlab.tasks.registry import load_env_cfg

    from ..tasks.tracking.rl import MyoSuiteMotionTrackingOnPolicyRunner
    from ..tasks.tracking.tracking_env_cfg import MyoSuiteTrackingEnvCfg
  except ImportError:
    return

  tracking_count = 0
  for env_id in myosuite_envs:
    tracking_task_id = (
      f"Mjlab-MyoSuite-Tracking-{env_id.split('/')[-1]}"
      if "/" in env_id
      else f"Mjlab-MyoSuite-Tracking-{env_id}"
    )
    try:
      load_env_cfg(tracking_task_id)
      continue
    except (KeyError, ValueError, AttributeError):
      pass

    def make_tracking_env_cfg(base_env_id=env_id, play=False):
      return MyoSuiteTrackingEnvCfg()

    def make_tracking_rl_cfg():
      return get_default_myosuite_rl_cfg()

    register_mjlab_task(
      task_id=tracking_task_id,
      env_cfg=make_tracking_env_cfg(play=False),  # type: ignore[arg-type]
      play_env_cfg=make_tracking_env_cfg(play=True),  # type: ignore[arg-type]
      rl_cfg=make_tracking_rl_cfg(),
      runner_cls=MyoSuiteMotionTrackingOnPolicyRunner,
    )
    tracking_count += 1

  if tracking_count:
    print(
      f"[INFO] Registered {tracking_count} MyoSuite tracking tasks (task_id: Mjlab-MyoSuite-Tracking-*)"
    )
