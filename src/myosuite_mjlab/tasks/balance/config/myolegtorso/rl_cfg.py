"""RL configuration for MyoLegsTorso standing-balance task."""

from dataclasses import replace

from myosuite_mjlab.tasks.velocity.myoskeleton.rl_cfg import (
    myoskeleton_ppo_runner_cfg,
)


def myolegtorso_balance_ppo_runner_cfg():
    """MyoLegsTorso balance PPO config: reuse MyoSkeleton PPO with new experiment name."""
    return replace(
        myoskeleton_ppo_runner_cfg(),
        experiment_name="myolegtorso_balance",
    )
