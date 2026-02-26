"""RL configuration for MyoLegTorso velocity task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from ..myoskeleton.rl_cfg import myoskeleton_ppo_runner_cfg


def myolegtorso_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """MyoLegTorso velocity PPO config."""
    cfg = myoskeleton_ppo_runner_cfg()
    cfg.experiment_name = "myolegtorso_velocity"
    return cfg
