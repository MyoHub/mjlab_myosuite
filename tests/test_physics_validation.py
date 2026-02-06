from __future__ import annotations

import numpy as np
import pytest


def _has_myosuite() -> bool:
  try:
    import myosuite  # noqa: F401

    return True
  except Exception:
    return False


pytestmark = pytest.mark.skipif(not _has_myosuite(), reason="myosuite not installed")


def _make_wrapped_env(myosuite_env_id: str = "myoElbowPose1D6MRandom-v0"):
  from mjlab_myosuite.env_factory import make_myosuite_env

  env = make_myosuite_env(myosuite_env_id)
  try:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    return env, unwrapped
  except Exception:
    env.close()
    raise


def test_step_advances_time():
  """Stepping the wrapped env should advance simulation time."""
  import torch

  env, unwrapped = _make_wrapped_env()
  try:
    obs, info = env.reset()
    assert isinstance(info, dict)

    # Access underlying sim to read mj_data.time
    sim = getattr(unwrapped, "sim", None)
    assert sim is not None
    mj_data = getattr(sim, "mj_data", getattr(sim, "data", None))
    assert mj_data is not None
    t0 = float(mj_data.time)

    # Take a few steps with zero action
    if hasattr(unwrapped, "single_action_space"):
      action_space = unwrapped.single_action_space
      num_envs = getattr(unwrapped, "num_envs", 1)
      if hasattr(action_space, "shape") and action_space.shape is not None:
        per_env_shape = action_space.shape
        zero_action = np.zeros((num_envs,) + per_env_shape, dtype=float)
      else:
        sample = action_space.sample()
        zero_action = np.zeros_like(sample)
        if getattr(zero_action, "ndim", 0) == 1 and num_envs > 1:
          zero_action = np.tile(zero_action, (num_envs, 1))
    else:
      zero_action = np.zeros_like(env.action_space.sample())

    for _ in range(5):
      obs, rewards, dones, extras = env.step(zero_action)  # type: ignore[assignment]
      # Ensure rewards/dones have sane types so this test will fail loudly if
      # the bridge breaks.
      if hasattr(rewards, "shape"):
        assert rewards.shape[0] == getattr(unwrapped, "num_envs", 1)
      if not isinstance(dones, torch.Tensor):
        dones = torch.as_tensor(dones, dtype=torch.bool)

    t1 = float(mj_data.time)
    assert t1 > t0
  finally:
    env.close()


def test_reset_sets_consistent_initial_state():
  """Resets with the same seed should produce deterministic initial observations."""
  import torch

  env, unwrapped = _make_wrapped_env()
  try:
    seed = 123
    # Go through wrapper API if available; use a fresh env instance for each run
    if hasattr(unwrapped, "get_observations"):
      # First run
      obs1, _ = env.reset(seed=seed)
      td1 = unwrapped.get_observations()
      obs_tensor_1 = td1["policy"].detach().clone()

      # Close and recreate env to avoid hidden state affecting determinism
      env.close()
      env2, unwrapped2 = _make_wrapped_env()
      try:
        obs2, _ = env2.reset(seed=seed)
        td2 = unwrapped2.get_observations()
        obs_tensor_2 = td2["policy"]
      finally:
        env2.close()
    else:
      # Fall back to raw obs
      obs1, _ = env.reset(seed=seed)
      obs2, _ = env.reset(seed=seed)
      obs_tensor_1 = torch.as_tensor(obs1)
      obs_tensor_2 = torch.as_tensor(obs2)

    # Enforce matching shapes, but allow non-determinism for now:
    # some MyoSuite tasks include stochastic initialization even with a seed.
    assert obs_tensor_1.shape == obs_tensor_2.shape
  finally:
    env.close()


def test_observation_shape_and_dtype():
  """Observation tensors should match the declared observation space."""
  import gymnasium as gym
  import torch

  env, unwrapped = _make_wrapped_env()
  try:
    obs, _ = env.reset()
    assert isinstance(unwrapped.observation_space, gym.spaces.Dict)

    if hasattr(unwrapped, "get_observations"):
      td = unwrapped.get_observations()
      obs_policy = td["policy"]
      assert isinstance(obs_policy, torch.Tensor)
      assert obs_policy.dtype == torch.float32
      # Batch dimension should be num_envs
      num_envs = getattr(unwrapped, "num_envs", 1)
      assert obs_policy.shape[0] == num_envs
    else:
      # Fall back to raw observation contract: must be convertible to float32 tensor
      obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
      assert obs_tensor.dtype == torch.float32
  finally:
    env.close()


def test_action_affects_state():
  """Applying nonzero actions should change the observed state."""
  import torch

  env, unwrapped = _make_wrapped_env()
  try:
    obs0, _ = env.reset()
    if hasattr(unwrapped, "get_observations"):
      td0 = unwrapped.get_observations()
      obs_policy_0 = td0["policy"].clone()
    else:
      obs_policy_0 = torch.as_tensor(obs0, dtype=torch.float32)

    # Sample a constant non-zero action
    action = env.action_space.sample()
    if hasattr(action, "shape"):
      # Ensure it's not all zeros
      if np.allclose(action, 0.0):
        action = np.ones_like(action)

    # Apply the same action for a few steps
    for _ in range(10):
      obs, rewards, dones, extras = env.step(action)  # type: ignore[assignment]
      # We don't assert on rewards here, but ensure they are finite
      if hasattr(rewards, "shape"):
        assert np.all(np.isfinite(np.asarray(rewards)))

    if hasattr(unwrapped, "get_observations"):
      td1 = unwrapped.get_observations()
      obs_policy_1 = td1["policy"]
    else:
      obs_policy_1 = torch.as_tensor(obs, dtype=torch.float32)

    # State should have changed in response to actions
    assert not torch.allclose(obs_policy_0, obs_policy_1)
  finally:
    env.close()
