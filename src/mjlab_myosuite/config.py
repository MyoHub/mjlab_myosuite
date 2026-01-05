"""Default configuration generators for MyoSuite environments."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  pass


@dataclass
class MyoSuiteEnvCfg:
  """Configuration for MyoSuite environments compatible with mjlab.

  This configuration follows mjlab's pattern for environment configs.
  For GPU-accelerated MyoSuite (mjx/warp versions), set device to "cuda:0".

  Example:
      >>> cfg = MyoSuiteEnvCfg()
      >>> cfg.num_envs = 4096  # For training
      >>> cfg.device = "cuda:0"  # Use GPU
      >>> env = gym.make("Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0", cfg=cfg)
  """

  num_envs: int = 1
  """Number of parallel environments. Use 4096+ for GPU training."""
  device: str = "cpu"
  """Device to use. Set to 'cuda:0' for GPU-accelerated MyoSuite (mjx/warp versions)."""


def get_default_myosuite_rl_cfg() -> Any:
  """Get default RL configuration for MyoSuite environments.

  Returns:
    Default RslRlOnPolicyRunnerCfg with reasonable defaults for MyoSuite tasks
  """
  # Lazy import to avoid triggering mjlab import chain
  from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
  )

  return RslRlOnPolicyRunnerCfg(
    experiment_name="myosuite",
    run_name="",
    max_iterations=1000,
    num_steps_per_env=24,
    policy=RslRlPpoActorCriticCfg(
      actor_hidden_dims=(256, 256),
      critic_hidden_dims=(256, 256),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      learning_rate=3e-4,
      num_learning_epochs=5,
      num_mini_batches=4,
      gamma=0.99,
      lam=0.95,
    ),
    clip_actions=None,  # Let MyoSuite handle action bounds
  )
