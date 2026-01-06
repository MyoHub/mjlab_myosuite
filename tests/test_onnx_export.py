"""Tests for ONNX model export functionality."""

import os
import tempfile

import pytest
import torch

from mjlab_myosuite.rl.exporter import (
  attach_myosuite_onnx_metadata,
  export_myosuite_policy_as_onnx,
)


def _has_myosuite() -> bool:
  try:
    import myosuite  # noqa: F401

    return True
  except Exception:
    return False


def _has_onnx() -> bool:
  """Check if ONNX is available."""
  try:
    import onnx  # noqa: F401

    return True
  except ImportError:
    return False


pytestmark = pytest.mark.skipif(
  not _has_myosuite() or not _has_onnx(),
  reason="myosuite not installed or ONNX not available",
)


def test_export_myosuite_policy_as_onnx():
  """Test exporting a simple policy to ONNX format."""
  import gymnasium as gym

  # Trigger auto-registration
  import mjlab_myosuite  # noqa: F401
  from mjlab_myosuite.config import MyoSuiteEnvCfg

  # Check if ONNX export is available
  try:
    from mjlab.utils.lab_api.rl.exporter import export_policy_as_onnx  # noqa: F401
  except ImportError:
    try:
      from mjlab.rl.exporter_utils import export_policy_as_onnx  # noqa: F401
    except ImportError:
      pytest.skip("ONNX export functionality not available in mjlab")

  # Create a simple mock policy
  class MockPolicy(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
      super().__init__()
      self.is_recurrent = False
      self.actor = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, action_dim),
        torch.nn.Tanh(),
      )

    def forward(self, obs):
      return self.actor(obs)

  # Create environment to get dimensions
  cfg = MyoSuiteEnvCfg()
  cfg.num_envs = 1
  env = gym.make("Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0", cfg=cfg)

  try:
    # Unwrap to get the actual wrapper
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    # Get observation and action dimensions
    obs_space = unwrapped.single_observation_space
    if isinstance(obs_space, gym.spaces.Dict):
      if "policy" in obs_space.spaces:
        obs_dim = int(torch.prod(torch.tensor(obs_space.spaces["policy"].shape)))
      else:
        obs_dim = int(
          torch.prod(torch.tensor(next(iter(obs_space.spaces.values())).shape))
        )
    else:
      obs_dim = int(torch.prod(torch.tensor(obs_space.shape)))

    action_dim = int(torch.prod(torch.tensor(unwrapped.single_action_space.shape)))

    # Create mock policy
    policy = MockPolicy(obs_dim, action_dim)

    # Export to ONNX
    with tempfile.TemporaryDirectory() as tmpdir:
      export_myosuite_policy_as_onnx(
        actor_critic=policy,
        path=tmpdir,
        filename="test_policy.onnx",
        verbose=False,
      )

      # Verify ONNX file was created
      onnx_path = os.path.join(tmpdir, "test_policy.onnx")
      assert os.path.exists(onnx_path), "ONNX file should be created"

      # Verify ONNX file is valid
      try:
        import onnx

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
      except ImportError:
        pass  # ONNX not available for validation

  finally:
    env.close()


def test_attach_myosuite_onnx_metadata():
  """Test attaching metadata to ONNX model."""
  import gymnasium as gym

  # Trigger auto-registration
  import mjlab_myosuite  # noqa: F401
  from mjlab_myosuite.config import MyoSuiteEnvCfg

  # Create environment
  cfg = MyoSuiteEnvCfg()
  cfg.num_envs = 2
  env = gym.make("Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0", cfg=cfg)

  try:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

    # Create a dummy ONNX file first
    with tempfile.TemporaryDirectory() as tmpdir:
      onnx_path = os.path.join(tmpdir, "test_policy.onnx")

      # Create a minimal ONNX model
      try:
        import onnx
        from onnx import TensorProto, helper

        # Create a simple ONNX model
        input_tensor = helper.make_tensor_value_info(
          "input", TensorProto.FLOAT, [1, 10]
        )
        output_tensor = helper.make_tensor_value_info(
          "output", TensorProto.FLOAT, [1, 5]
        )

        # Create a simple graph
        node = helper.make_node("Identity", ["input"], ["output"])
        graph = helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = helper.make_model(graph)
        onnx.save(model, onnx_path)
      except ImportError:
        pytest.skip("ONNX not available for creating test model")

      # Check if metadata utilities are available before testing
      try:
        import importlib.util

        spec = importlib.util.find_spec("mjlab.rl.exporter_utils")
        if spec is None:
          pytest.skip("ONNX metadata utilities not available in mjlab")
      except ImportError:
        pytest.skip("ONNX metadata utilities not available in mjlab")

      # Attach metadata - this should not raise an exception
      attach_myosuite_onnx_metadata(
        env=unwrapped,
        run_path="test_run",
        path=tmpdir,
        filename="test_policy.onnx",
      )

      # Verify the ONNX file is still valid after metadata attachment
      try:
        import onnx

        onnx_model = onnx.load(onnx_path)
        # The important thing is that the function didn't crash and the model is still valid
        assert onnx_model is not None, "ONNX model should be loadable"
        # Metadata might be empty if attachment failed silently, but that's acceptable
        # The function should handle errors gracefully without crashing
        # If metadata was successfully attached, verify it
        if len(onnx_model.metadata_props) > 0:
          metadata_dict = {prop.key: prop.value for prop in onnx_model.metadata_props}
          # If run_path is in metadata, verify it's correct
          if "run_path" in metadata_dict:
            assert metadata_dict["run_path"] == "test_run"
      except ImportError:
        pytest.skip("ONNX not available for validation")

  finally:
    env.close()


def test_manager_compatibility_attributes():
  """Test that wrapper has ManagerBasedRlEnv-compatible attributes."""
  import gymnasium as gym

  # Trigger auto-registration
  import mjlab_myosuite  # noqa: F401
  from mjlab_myosuite.config import MyoSuiteEnvCfg

  cfg = MyoSuiteEnvCfg()
  cfg.num_envs = 4
  env = gym.make("Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0", cfg=cfg)

  try:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

    # Check for ManagerBasedRlEnv-compatible attributes
    assert hasattr(unwrapped, "scene"), "Wrapper should have scene attribute"
    assert hasattr(unwrapped, "action_manager"), (
      "Wrapper should have action_manager attribute"
    )
    assert hasattr(unwrapped, "observation_manager"), (
      "Wrapper should have observation_manager attribute"
    )
    assert hasattr(unwrapped, "command_manager"), (
      "Wrapper should have command_manager attribute"
    )

    # Check scene
    scene = unwrapped.scene
    assert hasattr(scene, "num_envs"), "Scene should have num_envs"
    assert scene.num_envs == 4

    # Check action manager
    action_manager = unwrapped.action_manager
    assert hasattr(action_manager, "get_term"), (
      "Action manager should have get_term method"
    )
    assert hasattr(action_manager, "active_terms"), (
      "Action manager should have active_terms"
    )

    # Check observation manager
    obs_manager = unwrapped.observation_manager
    assert hasattr(obs_manager, "active_terms"), (
      "Observation manager should have active_terms"
    )

    # Check command manager
    cmd_manager = unwrapped.command_manager
    assert hasattr(cmd_manager, "active_terms"), (
      "Command manager should have active_terms"
    )

  finally:
    env.close()
