"""Task registration for MyoSkeleton velocity in mjlab."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import myoskeleton_flat_env_cfg, myoskeleton_standing_env_cfg
from .rl_cfg import myoskeleton_ppo_runner_cfg, myoskeleton_standing_ppo_runner_cfg

try:
    register_mjlab_task(
        task_id="Mjlab-Velocity-Flat-MyoSkeleton",
        env_cfg=myoskeleton_flat_env_cfg(),
        play_env_cfg=myoskeleton_flat_env_cfg(play=True),
        rl_cfg=myoskeleton_ppo_runner_cfg(),
        runner_cls=VelocityOnPolicyRunner,
    )
except ValueError:
    pass

try:
    register_mjlab_task(
        task_id="Mjlab-Standing-Flat-MyoSkeleton",
        env_cfg=myoskeleton_standing_env_cfg(),
        play_env_cfg=myoskeleton_standing_env_cfg(play=True),
        rl_cfg=myoskeleton_standing_ppo_runner_cfg(),
        runner_cls=VelocityOnPolicyRunner,
    )
except ValueError:
    pass
