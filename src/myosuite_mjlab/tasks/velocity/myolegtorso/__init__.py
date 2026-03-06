"""Task registration for MyoLegTorso velocity in mjlab."""

from mjlab.tasks.registry import register_mjlab_task

from myosuite_mjlab.rl import MyosuiteVelocityRunner

from .env_cfgs import myolegtorso_flat_env_cfg
from .rl_cfg import myolegtorso_ppo_runner_cfg

try:
    register_mjlab_task(
        task_id="MjlabMyoSuite-Velocity-Flat-MyoLegsTorso",
        env_cfg=myolegtorso_flat_env_cfg(),
        play_env_cfg=myolegtorso_flat_env_cfg(play=True),
        rl_cfg=myolegtorso_ppo_runner_cfg(),
        runner_cls=MyosuiteVelocityRunner,
    )
except ValueError:
    pass  # already registered
