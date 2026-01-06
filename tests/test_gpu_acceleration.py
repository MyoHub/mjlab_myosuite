"""Tests to verify GPU acceleration for MyoSuite environments."""

import pytest
import torch


def _has_myosuite() -> bool:
  try:
    import myosuite  # noqa: F401

    return True
  except Exception:
    return False


def _has_cuda() -> bool:
  """Check if CUDA is available."""
  return torch.cuda.is_available()


pytestmark = pytest.mark.skipif(
  not _has_myosuite() or not _has_cuda(),
  reason="myosuite not installed or CUDA not available",
)


def test_gpu_acceleration_observations():
  """Test that observations are on GPU when device is set to cuda:0."""
  import gymnasium as gym

  # Trigger auto-registration
  import mjlab_myosuite  # noqa: F401
  from mjlab_myosuite.config import MyoSuiteEnvCfg

  # Create environment with GPU device
  cfg = MyoSuiteEnvCfg()
  cfg.device = "cuda:0"
  cfg.num_envs = 4  # Use multiple envs to test batching

  env = gym.make("Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0", cfg=cfg)

  try:
    # Verify wrapper device is set correctly
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    assert hasattr(unwrapped, "device")
    assert str(unwrapped.device).startswith("cuda")

    # Reset and get observations
    obs, info = env.reset()
    assert isinstance(info, dict)

    # Check that observations are on GPU
    if hasattr(unwrapped, "get_observations"):
      td = unwrapped.get_observations()
      assert "policy" in td
      policy_obs = td["policy"]
      if isinstance(policy_obs, torch.Tensor):
        assert policy_obs.device.type == "cuda", (
          f"Policy observations should be on GPU, but found device: {policy_obs.device}"
        )
        assert policy_obs.device.index == 0, (
          f"Policy observations should be on cuda:0, but found: {policy_obs.device}"
        )

      if "critic" in td:
        critic_obs = td["critic"]
        if isinstance(critic_obs, torch.Tensor):
          assert critic_obs.device.type == "cuda", (
            f"Critic observations should be on GPU, but found device: {critic_obs.device}"
          )

    # Test step to verify rewards and done flags are on GPU
    action = env.action_space.sample()
    obs, rewards, dones, extras = env.step(action)  # type: ignore[assignment]
    # Extract terminated and truncated from extras if needed
    # Ensure they are torch tensors to avoid numpy array boolean ambiguity
    # Ensure dones is a tensor
    if not isinstance(dones, torch.Tensor):
      dones = torch.as_tensor(dones, dtype=torch.bool)
    terminated = extras.get("terminated", dones)
    if not isinstance(terminated, torch.Tensor):
      terminated = torch.as_tensor(terminated, dtype=torch.bool)
    # Get truncated from extras, or create zeros tensor if not available
    truncated_raw = extras.get("truncated", None)
    if truncated_raw is None:
      truncated = torch.zeros_like(dones, dtype=torch.bool)
    elif isinstance(truncated_raw, torch.Tensor):
      truncated = truncated_raw
    else:
      truncated = torch.as_tensor(truncated_raw, dtype=torch.bool)
    done = terminated | truncated

    # Verify reward is on GPU
    if isinstance(rewards, torch.Tensor):
      assert rewards.device.type == "cuda", (
        f"Reward should be on GPU, but found device: {rewards.device}"
      )

    # Verify done is on GPU
    if isinstance(done, torch.Tensor):
      assert done.device.type == "cuda", (
        f"Done flag should be on GPU, but found device: {done.device}"
      )

  finally:
    env.close()


def test_gpu_acceleration_simulation_data():
  """Test that simulation data structures are accessible and potentially on GPU for mjx/warp versions."""
  import gymnasium as gym

  # Trigger auto-registration
  import mjlab_myosuite  # noqa: F401
  from mjlab_myosuite.config import MyoSuiteEnvCfg

  # Create environment with GPU device
  cfg = MyoSuiteEnvCfg()
  cfg.device = "cuda:0"
  cfg.num_envs = 2

  env = gym.make("Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0", cfg=cfg)

  try:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    assert hasattr(unwrapped, "sim")

    sim = unwrapped.sim
    assert hasattr(sim, "mj_model") or hasattr(sim, "model")
    assert hasattr(sim, "mj_data") or hasattr(sim, "data") or hasattr(sim, "wp_data")

    # For mjx/warp versions, check if we can access GPU-accelerated data
    # The presence of 'data' or 'model' (instead of 'mj_data'/'mj_model') might indicate mjx/warp
    has_wp_data = hasattr(sim, "wp_data")

    # If we have wp_data, it might indicate GPU acceleration
    if has_wp_data:
      wp_data = sim.wp_data
      # wp_data might have device information
      if hasattr(wp_data, "device"):
        # Some versions might expose device information
        pass

    # Verify that the wrapper correctly handles GPU device
    assert str(unwrapped.device).startswith("cuda")

  finally:
    env.close()


def test_gpu_acceleration_batched_environments():
  """Test that batched environments work correctly with GPU acceleration."""
  import gymnasium as gym

  # Trigger auto-registration
  import mjlab_myosuite  # noqa: F401
  from mjlab_myosuite.config import MyoSuiteEnvCfg

  # Create environment with multiple parallel environments on GPU
  cfg = MyoSuiteEnvCfg()
  cfg.device = "cuda:0"
  cfg.num_envs = 8  # Use more envs to test batching

  env = gym.make("Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0", cfg=cfg)

  try:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    assert unwrapped.num_envs == 8
    assert str(unwrapped.device).startswith("cuda")

    # Reset and verify batch size
    obs, info = env.reset()
    assert isinstance(info, dict)

    # Get observations and verify batch size and device
    if hasattr(unwrapped, "get_observations"):
      td = unwrapped.get_observations()
      assert "policy" in td
      policy_obs = td["policy"]
      if isinstance(policy_obs, torch.Tensor):
        # Verify batch size
        assert policy_obs.shape[0] == 8, (
          f"Expected batch size 8, but got {policy_obs.shape[0]}"
        )
        # Verify device
        assert policy_obs.device.type == "cuda", (
          f"Policy observations should be on GPU, but found device: {policy_obs.device}"
        )

    # Test step with batched actions
    action = env.action_space.sample()
    obs, rewards, dones, extras = env.step(action)  # type: ignore[assignment]
    # Extract terminated and truncated from extras if needed
    # Ensure they are torch tensors to avoid numpy array boolean ambiguity
    # Ensure dones is a tensor
    if not isinstance(dones, torch.Tensor):
      dones = torch.as_tensor(dones, dtype=torch.bool)
    terminated = extras.get("terminated", dones)
    if not isinstance(terminated, torch.Tensor):
      terminated = torch.as_tensor(terminated, dtype=torch.bool)
    # Get truncated from extras, or create zeros tensor if not available
    truncated_raw = extras.get("truncated", None)
    if truncated_raw is None:
      truncated = torch.zeros_like(dones, dtype=torch.bool)
    elif isinstance(truncated_raw, torch.Tensor):
      truncated = truncated_raw
    else:
      truncated = torch.as_tensor(truncated_raw, dtype=torch.bool)
    done = terminated | truncated

    # Verify batch sizes
    if isinstance(rewards, torch.Tensor):
      assert rewards.shape[0] == 8
      assert rewards.device.type == "cuda"

    if isinstance(done, torch.Tensor):
      assert done.shape[0] == 8
      assert done.device.type == "cuda"

  finally:
    env.close()


def test_gpu_acceleration_via_factory():
  """Test GPU acceleration when creating environment via factory function."""
  from mjlab_myosuite.config import MyoSuiteEnvCfg
  from mjlab_myosuite.env_factory import make_myosuite_env

  # Create config with GPU device
  cfg = MyoSuiteEnvCfg()
  cfg.device = "cuda:0"
  cfg.num_envs = 4

  # Create environment with GPU device via factory
  wrapped = make_myosuite_env("myoElbowPose1D6MRandom-v0", cfg=cfg, num_envs=4)

  try:
    # Verify device is set correctly
    assert str(wrapped.device).startswith("cuda")
    assert wrapped.num_envs == 4

    # Reset and get observations
    obs, info = wrapped.reset()
    assert isinstance(info, dict)

    # Verify observations are on GPU
    if hasattr(wrapped, "get_observations"):
      td = wrapped.get_observations()
      assert "policy" in td
      policy_obs = td["policy"]
      if isinstance(policy_obs, torch.Tensor) and policy_obs.device is not None:
        assert policy_obs.device.type == "cuda", (
          f"Policy observations should be on GPU, but found device: {policy_obs.device}"
        )

    # Test step
    action = wrapped.action_space.sample()
    obs, rewards, dones, extras = wrapped.step(action)
    # Extract terminated and truncated from extras if needed
    # Ensure they are torch tensors to avoid numpy array boolean ambiguity
    terminated = extras.get("terminated", dones)
    if not isinstance(terminated, torch.Tensor):
      terminated = torch.as_tensor(terminated)
    # Get truncated from extras, or create zeros tensor if not available
    truncated_raw = extras.get("truncated", None)
    if truncated_raw is None:
      truncated = torch.zeros_like(dones, dtype=torch.bool)
    elif isinstance(truncated_raw, torch.Tensor):
      truncated = truncated_raw
    else:
      truncated = torch.as_tensor(truncated_raw, dtype=torch.bool)
    _ = terminated | truncated  # Check done flag

    # Verify reward is on GPU
    if isinstance(rewards, torch.Tensor):
      assert rewards.device.type == "cuda"

  finally:
    wrapped.close()
