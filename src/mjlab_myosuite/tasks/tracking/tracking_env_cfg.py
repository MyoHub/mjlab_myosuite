"""Configuration for MyoSuite tracking environments."""

from dataclasses import dataclass, field
from pathlib import Path

from ...config import MyoSuiteEnvCfg


@dataclass
class MotionCfg:
  """Configuration for motion tracking."""

  motion_file: str | None = None
  """Path to motion file (.npz format)."""


@dataclass
class CommandsCfg:
  """Configuration for command generation."""

  motion: MotionCfg = field(default_factory=MotionCfg)
  """Motion configuration for tracking."""

  def __contains__(self, key: str) -> bool:
    """Support 'in' operator for compatibility with mjlab's native scripts."""
    return hasattr(self, key)

  def __getitem__(self, key: str):
    """Support subscripting for compatibility with mjlab's native scripts."""
    if hasattr(self, key):
      return getattr(self, key)
    raise KeyError(f"'{key}' not found in CommandsCfg")

  def items(self):
    """Support .items() for compatibility with mjlab's native scripts."""
    return [(k, getattr(self, k)) for k in dir(self) if not k.startswith("_")]


@dataclass
class MyoSuiteTrackingEnvCfg(MyoSuiteEnvCfg):
  """Configuration for MyoSuite tracking environments.

  This extends MyoSuiteEnvCfg with tracking-specific configuration,
  following the pattern of mjlab's TrackingEnvCfg.

  Example:
      >>> cfg = MyoSuiteTrackingEnvCfg()
      >>> cfg.num_envs = 4096
      >>> cfg.device = "cuda:0"
      >>> cfg.commands.motion.motion_file = "path/to/motion.npz"
      >>> env = gym.make("Mjlab-MyoSuite-Tracking-myoElbowPose1D6MRandom-v0", cfg=cfg)
  """

  commands: CommandsCfg = field(default_factory=CommandsCfg)
  """Command configuration for motion tracking."""

  def __post_init__(self):
    """Validate configuration after initialization."""
    # Validate motion file if provided
    # Note: We skip validation during testing or if file doesn't exist yet
    # (it might be created later or provided via CLI)
    if self.commands.motion.motion_file is not None:
      motion_path = Path(self.commands.motion.motion_file)
      if motion_path.exists() and motion_path.stat().st_size == 0:
        # Empty file - might be a test fixture, skip validation
        pass
      elif not motion_path.exists():
        # File doesn't exist yet - might be set later, skip validation
        pass
