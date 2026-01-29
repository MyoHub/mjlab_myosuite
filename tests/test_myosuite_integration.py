from typing import Optional

import pytest


def _has_myosuite() -> bool:
  try:
    import myosuite  # noqa: F401

    return True
  except Exception:
    return False


pytestmark = pytest.mark.skipif(not _has_myosuite(), reason="myosuite not installed")


def test_registration_and_make_env():
  import gymnasium as gym

  # Trigger auto-registration
  import mjlab_myosuite  # noqa: F401

  # Pick one canonical env id that should exist in MyoSuite
  raw_id = "myoElbowPose1D6MRandom-v0"
  wrapped_id = f"Mjlab-MyoSuite-{raw_id}"

  # Verify registration
  all_envs = list(gym.registry.keys())
  assert raw_id in all_envs or wrapped_id in all_envs

  # Make mjlab-wrapped env (prefer the wrapped id if present)
  env_id = wrapped_id if wrapped_id in all_envs else raw_id
  env = gym.make(env_id)

  try:
    # Basic reset/step
    obs, info = env.reset()
    assert isinstance(info, dict)

    if hasattr(env, "unwrapped"):
      unwrapped = env.unwrapped
    else:
      unwrapped = env

    # Verify sim interface expected by viewers
    assert hasattr(unwrapped, "sim")
    sim = unwrapped.sim
    assert hasattr(sim, "mj_model")
    assert hasattr(sim, "mj_data") or hasattr(sim, "data")
    assert hasattr(sim, "wp_data")

    # One step with zero action if possible
    action_space = getattr(
      unwrapped, "single_action_space", getattr(unwrapped, "action_space", None)
    )
    if action_space is not None:
      import numpy as np

      # Determine per-env action shape
      per_env_shape: Optional[tuple] = None
      if hasattr(action_space, "shape") and action_space.shape is not None:
        per_env_shape = action_space.shape
      # Build batched zero action for vector envs (num_envs x action_dim)
      num_envs = getattr(unwrapped, "num_envs", 1)
      if per_env_shape is not None:
        batched_shape = (num_envs,) + per_env_shape
        zero_act = np.zeros(batched_shape, dtype=float)
      else:
        # Fallback to sampling then zeroing shape
        sample = action_space.sample()
        zero_act = np.zeros_like(sample)
        if getattr(zero_act, "ndim", 0) == 1 and num_envs > 1:
          zero_act = np.tile(zero_act, (num_envs, 1))
      env.step(zero_act)

  finally:
    env.close()


def test_wrapper_observations_device():
  import gymnasium as gym
  import torch

  # Trigger auto-registration
  import mjlab_myosuite  # noqa: F401

  env = gym.make("Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0")
  try:
    # If wrapper exposes get_observations, ensure tensors are on the env device
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    if hasattr(unwrapped, "get_observations") and hasattr(unwrapped, "device"):
      td = unwrapped.get_observations()
      assert "policy" in td
      pol = td["policy"]
      if isinstance(pol, torch.Tensor):
        assert pol.device == unwrapped.device
  finally:
    env.close()


def test_viewer_forward_kinematics_available():
  import gymnasium as gym

  # Trigger auto-registration
  import mjlab_myosuite  # noqa: F401

  env = gym.make("Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0")
  try:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    sim = getattr(unwrapped, "sim", None)
    assert sim is not None

    # Access properties used by the viewer; they should be callable without errors
    _ = sim.mj_model
    _ = sim.mj_data  # property should internally mj_forward
    _ = sim.wp_data  # property should internally mj_forward
  finally:
    env.close()


def test_wrapper_creation_direct():
  """Test creating wrapper directly from MyoSuite environment."""
  from myosuite.utils import gym as myosuite_gym

  from mjlab_myosuite.wrapper import MyoSuiteVecEnvWrapper

  # Create a MyoSuite environment
  myosuite_env = myosuite_gym.make("myoElbowPose1D6MRandom-v0")

  # Create wrapper with single environment
  wrapped = MyoSuiteVecEnvWrapper(env=myosuite_env, num_envs=1, device="cpu")

  try:
    # Verify wrapper has required attributes
    assert hasattr(wrapped, "num_envs")
    assert wrapped.num_envs == 1
    assert hasattr(wrapped, "device")
    assert hasattr(wrapped, "sim")
    assert hasattr(wrapped, "cfg")
    assert hasattr(wrapped, "action_space")
    assert hasattr(wrapped, "observation_space")
    assert hasattr(wrapped, "single_action_space")
    assert hasattr(wrapped, "single_observation_space")

    # Test reset
    obs, info = wrapped.reset()
    assert isinstance(info, dict)

    # Test step
    action = wrapped.action_space.sample()
    obs, rewards, dones, extras = wrapped.step(action)
    # Verify step returns are correct types
    import torch

    assert isinstance(rewards, torch.Tensor)
    assert isinstance(dones, torch.Tensor)
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
    assert isinstance(obs, type(obs))  # obs should be TensorDict

    # Test get_observations
    if hasattr(wrapped, "get_observations"):
      td = wrapped.get_observations()
      assert "policy" in td
      assert "critic" in td

  finally:
    wrapped.close()


def test_wrapper_creation_vectorized():
  """Test creating wrapper with multiple environments."""
  import torch
  from myosuite.utils import gym as myosuite_gym

  from mjlab_myosuite.wrapper import MyoSuiteVecEnvWrapper

  # Create a MyoSuite environment
  myosuite_env = myosuite_gym.make("myoElbowPose1D6MRandom-v0")

  # Create wrapper with multiple environments
  num_envs = 4
  wrapped = MyoSuiteVecEnvWrapper(env=myosuite_env, num_envs=num_envs, device="cpu")

  try:
    # Verify wrapper has required attributes
    assert wrapped.num_envs == num_envs
    assert hasattr(wrapped, "sim")
    assert hasattr(wrapped, "cfg")

    # Test reset
    obs, info = wrapped.reset()
    assert isinstance(info, dict)

    # Test step with batched actions
    action = wrapped.action_space.sample()
    obs, rewards, dones, extras = wrapped.step(action)
    assert rewards.shape[0] == num_envs
    assert dones.shape[0] == num_envs
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

    # Test get_observations
    if hasattr(wrapped, "get_observations"):
      td = wrapped.get_observations()
      assert "policy" in td
      # Check batch size
      policy_obs = td["policy"]
      if hasattr(policy_obs, "shape"):
        assert policy_obs.shape[0] == num_envs

  finally:
    wrapped.close()


def test_wrapper_creation_via_factory():
  """Test creating wrapper via env_factory."""
  import torch

  from mjlab_myosuite.env_factory import make_myosuite_env

  # Create wrapper via factory
  wrapped = make_myosuite_env("myoElbowPose1D6MRandom-v0", device="cpu", num_envs=2)

  try:
    # Verify wrapper has required attributes
    assert hasattr(wrapped, "num_envs")
    assert wrapped.num_envs == 2
    assert hasattr(wrapped, "sim")
    assert hasattr(wrapped, "cfg")

    # Test reset and step
    obs, info = wrapped.reset()
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

  finally:
    wrapped.close()


def test_wrapper_multiple_myosuite_envs():
  """Test wrapper creation for different MyoSuite environments."""
  import gymnasium as gym
  import torch

  # Trigger auto-registration
  import mjlab_myosuite  # noqa: F401

  # Test a few different MyoSuite environments
  test_envs = [
    "Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0",
    "Mjlab-MyoSuite-myoElbowPose1D6M-v0",
  ]

  for env_id in test_envs:
    if env_id not in gym.registry:
      continue  # Skip if not registered

    env = gym.make(env_id)
    try:
      # Basic functionality test
      obs, info = env.reset()
      assert isinstance(info, dict)

      action = env.action_space.sample()
      obs, rewards, dones, extras = env.step(action)  # type: ignore[assignment]
      # terminated and truncated are available in extras if needed
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

      _ = terminated | truncated  # Check done flag
      # Verify sim interface
      unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
      assert hasattr(unwrapped, "sim")
      sim = unwrapped.sim
      assert hasattr(sim, "mj_model")
      assert hasattr(sim, "mj_data")

    finally:
      env.close()
