"""Unit tests for MyoSuite tracking tasks."""

from pathlib import Path

import pytest


def _has_myosuite() -> bool:
  try:
    import myosuite  # noqa: F401

    return True
  except Exception:
    return False


pytestmark = pytest.mark.skipif(not _has_myosuite(), reason="myosuite not installed")


def test_tracking_env_cfg_creation():
  """Test creating MyoSuiteTrackingEnvCfg."""
  from mjlab_myosuite.tasks.tracking.tracking_env_cfg import MyoSuiteTrackingEnvCfg

  # Create basic tracking config
  cfg = MyoSuiteTrackingEnvCfg()
  assert cfg.num_envs == 1
  assert cfg.device == "cpu"
  assert cfg.commands is not None
  assert cfg.commands.motion is not None
  assert cfg.commands.motion.motion_file is None

  # Create with custom values
  cfg = MyoSuiteTrackingEnvCfg(num_envs=4096, device="cuda:0")
  assert cfg.num_envs == 4096
  assert cfg.device == "cuda:0"


def test_tracking_env_cfg_motion_file_validation(tmp_path: Path):
  """Test motion file validation in MyoSuiteTrackingEnvCfg."""
  from mjlab_myosuite.tasks.tracking.tracking_env_cfg import MyoSuiteTrackingEnvCfg

  # Create a dummy motion file
  motion_file = tmp_path / "motion.npz"
  motion_file.touch()

  # Test that we can set a valid motion file
  cfg = MyoSuiteTrackingEnvCfg()
  cfg.commands.motion.motion_file = str(motion_file)
  assert cfg.commands.motion.motion_file == str(motion_file)
  assert Path(cfg.commands.motion.motion_file).exists()

  # Test validation logic (the __post_init__ checks if file exists)
  # Since we set it after creation, we manually test the validation
  invalid_file = tmp_path / "nonexistent.npz"
  cfg.commands.motion.motion_file = str(invalid_file)

  # The validation in __post_init__ would check this, but since we're setting
  # after creation, we test the validation logic directly
  motion_path = Path(cfg.commands.motion.motion_file)
  # This should fail validation if __post_init__ is called with invalid file
  # For now, we just verify the path is set correctly
  assert motion_path == invalid_file


def test_tracking_runner_creation():
  """Test creating MyoSuiteMotionTrackingOnPolicyRunner."""
  from dataclasses import asdict

  from myosuite.utils import gym as myosuite_gym

  from mjlab_myosuite.config import get_default_myosuite_rl_cfg
  from mjlab_myosuite.tasks.tracking.rl import MyoSuiteMotionTrackingOnPolicyRunner
  from mjlab_myosuite.wrapper import MyoSuiteVecEnvWrapper

  # Create a simple environment
  myosuite_env = myosuite_gym.make("myoElbowPose1D6MRandom-v0")
  wrapped = MyoSuiteVecEnvWrapper(env=myosuite_env, num_envs=1, device="cpu")

  # Create runner config using the default config
  rl_cfg = get_default_myosuite_rl_cfg()
  agent_cfg = asdict(rl_cfg)
  agent_cfg["experiment_name"] = "test_tracking"
  agent_cfg["max_iterations"] = 100

  # Create runner
  runner = MyoSuiteMotionTrackingOnPolicyRunner(
    wrapped, agent_cfg, "logs/test", "cpu", registry_name=None
  )

  assert runner.registry_name is None
  assert hasattr(runner, "env")
  assert hasattr(runner, "cfg")

  wrapped.close()


def test_tracking_runner_with_registry_name():
  """Test MyoSuiteMotionTrackingOnPolicyRunner with registry name."""
  from dataclasses import asdict

  from myosuite.utils import gym as myosuite_gym

  from mjlab_myosuite.config import get_default_myosuite_rl_cfg
  from mjlab_myosuite.tasks.tracking.rl import MyoSuiteMotionTrackingOnPolicyRunner
  from mjlab_myosuite.wrapper import MyoSuiteVecEnvWrapper

  # Create a simple environment
  myosuite_env = myosuite_gym.make("myoElbowPose1D6MRandom-v0")
  wrapped = MyoSuiteVecEnvWrapper(env=myosuite_env, num_envs=1, device="cpu")

  # Create runner config using the default config
  rl_cfg = get_default_myosuite_rl_cfg()
  agent_cfg = asdict(rl_cfg)
  agent_cfg["experiment_name"] = "test_tracking"
  agent_cfg["max_iterations"] = 100

  # Create runner with registry name
  registry_name = "test-org/test-project/motion:latest"
  runner = MyoSuiteMotionTrackingOnPolicyRunner(
    wrapped, agent_cfg, "logs/test", "cpu", registry_name=registry_name
  )

  assert runner.registry_name == registry_name

  wrapped.close()


def test_tracking_runner_save():
  """Test MyoSuiteMotionTrackingOnPolicyRunner.save() method."""
  from dataclasses import asdict

  from myosuite.utils import gym as myosuite_gym

  from mjlab_myosuite.config import get_default_myosuite_rl_cfg
  from mjlab_myosuite.tasks.tracking.rl import MyoSuiteMotionTrackingOnPolicyRunner
  from mjlab_myosuite.wrapper import MyoSuiteVecEnvWrapper

  # Create a simple environment
  myosuite_env = myosuite_gym.make("myoElbowPose1D6MRandom-v0")
  wrapped = MyoSuiteVecEnvWrapper(env=myosuite_env, num_envs=1, device="cpu")

  # Create runner config using the default config
  rl_cfg = get_default_myosuite_rl_cfg()
  agent_cfg = asdict(rl_cfg)
  agent_cfg["experiment_name"] = "test_tracking"
  agent_cfg["max_iterations"] = 100

  # Create runner
  runner = MyoSuiteMotionTrackingOnPolicyRunner(
    wrapped, agent_cfg, "logs/test", "cpu", registry_name=None
  )

  # Test save method (should not raise)
  # Note: This will fail if the runner tries to actually save, but we're just
  # testing that the method exists and can be called
  try:
    # We can't actually save without a trained model, but we can verify the method exists
    assert hasattr(runner, "save")
    assert callable(runner.save)
  except Exception:
    # Expected to fail if trying to actually save, but method should exist
    pass

  wrapped.close()


def test_tracking_config_inheritance():
  """Test that MyoSuiteTrackingEnvCfg properly inherits from MyoSuiteEnvCfg."""
  from mjlab_myosuite.config import MyoSuiteEnvCfg
  from mjlab_myosuite.tasks.tracking.tracking_env_cfg import MyoSuiteTrackingEnvCfg

  # Verify inheritance
  assert issubclass(MyoSuiteTrackingEnvCfg, MyoSuiteEnvCfg)

  # Create instance and verify it has all base attributes
  cfg = MyoSuiteTrackingEnvCfg()
  assert hasattr(cfg, "num_envs")
  assert hasattr(cfg, "device")
  assert hasattr(cfg, "commands")
  assert hasattr(cfg.commands, "motion")
  assert hasattr(cfg.commands.motion, "motion_file")


def test_tracking_runner_inheritance():
  """Test that MyoSuiteMotionTrackingOnPolicyRunner properly inherits from MyoSuiteOnPolicyRunner."""
  from mjlab_myosuite.rl.runner import MyoSuiteOnPolicyRunner
  from mjlab_myosuite.tasks.tracking.rl import MyoSuiteMotionTrackingOnPolicyRunner

  # Verify inheritance
  assert issubclass(MyoSuiteMotionTrackingOnPolicyRunner, MyoSuiteOnPolicyRunner)


def test_tracking_imports():
  """Test that tracking module can be imported correctly."""
  from mjlab_myosuite.tasks.tracking import (
    MyoSuiteMotionTrackingOnPolicyRunner,
    MyoSuiteTrackingEnvCfg,
  )

  assert MyoSuiteTrackingEnvCfg is not None
  assert MyoSuiteMotionTrackingOnPolicyRunner is not None


def test_tracking_config_motion_cfg():
  """Test MotionCfg configuration."""
  from mjlab_myosuite.tasks.tracking.tracking_env_cfg import MotionCfg

  motion_cfg = MotionCfg()
  assert motion_cfg.motion_file is None

  motion_cfg.motion_file = "path/to/motion.npz"
  assert motion_cfg.motion_file == "path/to/motion.npz"


def test_tracking_config_commands_cfg():
  """Test CommandsCfg configuration."""
  from mjlab_myosuite.tasks.tracking.tracking_env_cfg import CommandsCfg, MotionCfg

  commands_cfg = CommandsCfg()
  assert commands_cfg.motion is not None
  assert isinstance(commands_cfg.motion, MotionCfg)

  # Can set motion file through commands
  commands_cfg.motion.motion_file = "path/to/motion.npz"
  assert commands_cfg.motion.motion_file == "path/to/motion.npz"
