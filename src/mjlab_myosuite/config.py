"""Default configuration generators for MyoSuite environments."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  pass


# Lazy import to avoid circular dependencies
def _get_manager_based_cfg_base():
  """Get ManagerBasedRlEnvCfg base class."""
  try:
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.managers import ActionTermCfg, ObservationGroupCfg
    from mjlab.scene import SceneCfg

    return ManagerBasedRlEnvCfg, SceneCfg, ActionTermCfg, ObservationGroupCfg
  except ImportError:
    # Fallback if mjlab not available
    return None, None, None, None


@dataclass
class MyoSuiteEnvCfg:
  """Configuration for MyoSuite environments compatible with mjlab.

  This configuration follows mjlab's pattern for environment configs.
  For GPU-accelerated MyoSuite (mjx/warp versions), set device to "cuda:0".

  Note: This config includes minimal attributes required by ManagerBasedRlEnvCfg
  for compatibility with mjlab's native scripts, but MyoSuite environments are
  actually created via gym.make(), not ManagerBasedRlEnv directly.

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
  commands: Any = None
  """Command configuration for tracking tasks. None for non-tracking tasks.

  This attribute exists for compatibility with mjlab's native scripts which check
  for commands to determine if a task is a tracking task. For tracking tasks,
  use MyoSuiteTrackingEnvCfg instead.
  """

  # Minimal attributes required by ManagerBasedRlEnvCfg for compatibility
  # These are stubs - MyoSuite environments don't actually use these
  # MyoSuite tasks are routed to custom training logic that uses gym.make()
  decimation: int = 1
  """Decimation (stub for ManagerBasedRlEnvCfg compatibility)."""
  scene: Any = None
  """Scene config (stub for ManagerBasedRlEnvCfg compatibility)."""
  observations: dict[str, Any] = field(default_factory=dict)
  """Observations (stub for ManagerBasedRlEnvCfg compatibility)."""
  actions: dict[str, Any] = field(default_factory=dict)
  """Actions (stub for ManagerBasedRlEnvCfg compatibility)."""
  events: dict[str, Any] = field(default_factory=dict)
  """Events (stub for ManagerBasedRlEnvCfg compatibility)."""
  rewards: dict[str, Any] = field(default_factory=dict)
  """Rewards (stub for ManagerBasedRlEnvCfg compatibility)."""
  terminations: dict[str, Any] = field(default_factory=dict)
  """Terminations (stub for ManagerBasedRlEnvCfg compatibility)."""
  seed: int | None = None
  """Random seed."""
  sim: Any = field(default=None)
  """Simulation config (stub for ManagerBasedRlEnvCfg compatibility)."""
  viewer: Any = field(default=None)
  """Viewer config (stub for ManagerBasedRlEnvCfg compatibility)."""

  def __post_init__(self):
    """Initialize sim and viewer if they're None."""
    # Call parent __post_init__ if it exists
    if hasattr(super(), "__post_init__"):
      try:
        super().__post_init__()
      except AttributeError:
        pass  # Parent might not have __post_init__

    # Ensure sim is set to a proper SimulationCfg if None
    # Use object.__setattr__ for frozen dataclasses
    if self.sim is None:
      try:
        from mjlab.sim.sim import SimulationCfg

        object.__setattr__(self, "sim", SimulationCfg())
      except ImportError:
        # mjlab not available, create a minimal mock
        class _MockSimCfg:
          def __init__(self):
            self.mujoco = None  # Will be set when needed

        object.__setattr__(self, "sim", _MockSimCfg())

    # Ensure viewer is set to a proper ViewerConfig if None
    if self.viewer is None:
      try:
        from mjlab.viewer import ViewerConfig

        object.__setattr__(self, "viewer", ViewerConfig())
      except ImportError:
        # mjlab not available, create a minimal mock
        class _MockViewerCfg:
          def __init__(self):
            self.env_idx = 0  # Default to first environment
            self.height = 480
            self.width = 640

        object.__setattr__(self, "viewer", _MockViewerCfg())

  episode_length_s: float = 0.0
  """Episode length in seconds (stub for ManagerBasedRlEnvCfg compatibility)."""
  is_finite_horizon: bool = False
  """Finite horizon flag (stub for ManagerBasedRlEnvCfg compatibility)."""
  scale_rewards_by_dt: bool = True
  """Scale rewards by dt (stub for ManagerBasedRlEnvCfg compatibility)."""
  curriculum: Any = None
  """Curriculum configuration (stub for ManagerBasedRlEnvCfg compatibility)."""
  task_id: str | None = None
  """Task ID for ManagerBasedRlEnv integration (set automatically)."""


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
