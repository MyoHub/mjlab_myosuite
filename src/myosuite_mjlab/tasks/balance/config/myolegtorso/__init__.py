"""Task registration for MyoLegsTorso balance in mjlab."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import myolegtorso_balance_env_cfg
from .rl_cfg import myolegtorso_balance_ppo_runner_cfg

try:
    register_mjlab_task(
        task_id="Mjlab-Balance-Flat-MyoLegsTorso",
        env_cfg=myolegtorso_balance_env_cfg(),
        play_env_cfg=myolegtorso_balance_env_cfg(play=True),
        rl_cfg=myolegtorso_balance_ppo_runner_cfg(),
        runner_cls=VelocityOnPolicyRunner,
    )
except ValueError:
    pass  # already registered
