"""Task registration for MyoSkeleton velocity in mjlab."""

from mjlab.tasks.registry import register_mjlab_task

from myosuite_mjlab.rl import MyosuiteVelocityRunner

from .env_cfgs import myoskeleton_flat_env_cfg, myoskeleton_standing_env_cfg
from .rl_cfg import (
    myoskeleton_ppo_runner_cfg,
    myoskeleton_standing_ppo_runner_cfg,
)

try:
    register_mjlab_task(
        task_id="MjlabMyoSuite-Velocity-Flat-MyoSkeleton",
        env_cfg=myoskeleton_flat_env_cfg(),
        play_env_cfg=myoskeleton_flat_env_cfg(play=True),
        rl_cfg=myoskeleton_ppo_runner_cfg(),
        runner_cls=MyosuiteVelocityRunner,
    )
except ValueError:
    pass

try:
    register_mjlab_task(
        task_id="MjlabMyoSuite-Standing-Flat-MyoSkeleton",
        env_cfg=myoskeleton_standing_env_cfg(),
        play_env_cfg=myoskeleton_standing_env_cfg(play=True),
        rl_cfg=myoskeleton_standing_ppo_runner_cfg(),
        runner_cls=MyosuiteVelocityRunner,
    )
except ValueError:
    pass
