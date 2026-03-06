"""Task registration for MyoLeg velocity in mjlab."""

from mjlab.tasks.registry import register_mjlab_task

from myosuite_mjlab.rl import MyosuiteVelocityRunner

from .env_cfgs import myoleg_flat_env_cfg
from .rl_cfg import myoleg_ppo_runner_cfg

try:
    register_mjlab_task(
        "MjlabMyoSuite-Velocity-Flat-MyoLeg",
        env_cfg=myoleg_flat_env_cfg(),
        play_env_cfg=myoleg_flat_env_cfg(play=True),
        rl_cfg=myoleg_ppo_runner_cfg(),
        runner_cls=MyosuiteVelocityRunner,
    )
except ValueError:
    pass  # already registered
