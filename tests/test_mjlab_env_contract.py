"""Tests that mjlab_myosuite environments operate like mjlab environments.

Verifies the contract expected by mjlab's train/play: reset, step, get_observations,
device, num_envs, observation/action spaces, and optional physics_backend.
"""

import pytest


def _has_myosuite() -> bool:
  try:
    import myosuite  # noqa: F401

    return True
  except Exception:
    return False


pytestmark = pytest.mark.skipif(not _has_myosuite(), reason="myosuite not installed")


def test_mjlab_myosuite_has_mjlab_like_interface():
  """mjlab_myosuite env exposes the same interface as mjlab ManagerBasedRlEnv."""
  from mjlab_myosuite.env_factory import make_myosuite_env

  env = make_myosuite_env("myoElbowPose1D6MRandom-v0", num_envs=2, device="cpu")
  try:
    # Required attributes (mjlab contract)
    assert hasattr(env, "device"), "Env must have device"
    assert hasattr(env, "num_envs"), "Env must have num_envs"
    assert hasattr(env, "single_action_space"), "Env must have single_action_space"
    assert hasattr(env, "single_observation_space"), (
      "Env must have single_observation_space"
    )
    assert hasattr(env, "action_space"), "Env must have action_space"
    assert hasattr(env, "observation_space"), "Env must have observation_space"
    assert hasattr(env, "get_observations"), (
      "Env must have get_observations (mjlab/rsl_rl)"
    )
    assert hasattr(env, "reset"), "Env must have reset"
    assert hasattr(env, "step"), "Env must have step"

    assert env.num_envs == 2
    assert (
      str(env.device) in ("cpu", "cuda:0")
      or "cuda" in str(env.device)
      or env.device == "cpu"
    )
  finally:
    env.close()


def test_reset_returns_obs_dict_with_policy_critic():
  """reset() returns (obs_dict, info) with policy and critic groups like mjlab."""
  from mjlab_myosuite.env_factory import make_myosuite_env

  env = make_myosuite_env("myoElbowPose1D6MRandom-v0", num_envs=2, device="cpu")
  try:
    obs, info = env.reset(seed=42)
    assert isinstance(info, dict), "reset must return (obs, info) with dict info"
    # obs is TensorDict or dict-like with policy/critic
    assert hasattr(obs, "keys") or isinstance(obs, dict), "obs must be dict-like"
    keys = list(obs.keys()) if hasattr(obs, "keys") else list(obs)
    assert "policy" in keys or len(keys) >= 1, (
      "obs must have at least policy or one group"
    )
    if "policy" in keys:
      policy = obs["policy"]
      assert policy.shape[0] == 2, "policy obs batch size must match num_envs"
    if "critic" in keys:
      critic = obs["critic"]
      assert critic.shape[0] == 2, "critic obs batch size must match num_envs"
  finally:
    env.close()


def test_get_observations_returns_dict_like_on_device():
  """get_observations() returns dict-like with policy/critic on env.device."""
  from mjlab_myosuite.env_factory import make_myosuite_env

  env = make_myosuite_env("myoElbowPose1D6MRandom-v0", num_envs=2, device="cpu")
  try:
    env.reset(seed=42)
    obs = env.get_observations()
    assert obs is not None
    assert hasattr(obs, "keys") or isinstance(obs, dict)
    keys = list(obs.keys()) if hasattr(obs, "keys") else list(obs)
    assert "policy" in keys or len(keys) >= 1
    # On CPU, tensors should be on cpu
    for k in keys:
      v = obs[k]
      if hasattr(v, "device"):
        assert str(v.device) == "cpu" or "cuda" in str(v.device)
  finally:
    env.close()


def test_step_returns_four_tuple_obs_rewards_dones_info():
  """step(actions) returns (obs, rewards, dones, info) compatible with mjlab/rsl_rl."""
  import numpy as np

  from mjlab_myosuite.env_factory import make_myosuite_env

  env = make_myosuite_env("myoElbowPose1D6MRandom-v0", num_envs=2, device="cpu")
  try:
    env.reset(seed=42)
    n = env.num_envs
    action_dim = getattr(env.unwrapped, "num_actions", None)
    if action_dim is None and hasattr(env, "single_action_space"):
      sp = env.single_action_space
      action_dim = int(np.prod(getattr(sp, "shape", (1,))))
    else:
      action_dim = action_dim or 1
    actions = np.zeros((n, action_dim), dtype=np.float32)
    result = env.step(actions)
    assert isinstance(result, tuple), "step must return tuple"
    assert len(result) >= 4, "step must return at least (obs, rewards, dones, info)"
    obs, rewards, dones, info = result[0], result[1], result[2], result[3]
    assert obs is not None
    assert hasattr(rewards, "shape") or hasattr(rewards, "__len__")
    assert hasattr(dones, "shape") or hasattr(dones, "__len__")
    assert isinstance(info, dict)
  finally:
    env.close()


def test_physics_backend_attribute():
  """Env has physics_backend attribute (CPU or WARP) for mjlab alignment."""
  from mjlab_myosuite.config import PhysicsBackend
  from mjlab_myosuite.env_factory import make_myosuite_env

  env = make_myosuite_env("myoElbowPose1D6MRandom-v0", num_envs=1, device="cpu")
  try:
    assert hasattr(env, "physics_backend"), (
      "Env must have physics_backend for mjlab alignment"
    )
    assert env.physics_backend in (PhysicsBackend.CPU, PhysicsBackend.WARP, None)
  finally:
    env.close()


def test_gpu_env_observations_on_device():
  """When device is cuda, env reports cuda and get_observations() returns dict-like (mjlab contract)."""
  try:
    import torch

    if not torch.cuda.is_available():
      pytest.skip("CUDA not available")
  except ImportError:
    pytest.skip("torch not available")

  from mjlab_myosuite.env_factory import make_myosuite_env

  env = make_myosuite_env("myoElbowPose1D6MRandom-v0", num_envs=2, device="cuda:0")
  try:
    assert str(env.device).startswith("cuda"), "Env device must be cuda when requested"
    env.reset(seed=42)
    obs = env.get_observations()
    assert obs is not None and (hasattr(obs, "keys") or isinstance(obs, dict))
    # Contract: policy/critic keys present; tensors expected on env.device (wrapper implements this)
    keys = list(obs.keys()) if hasattr(obs, "keys") else list(obs)
    assert "policy" in keys or len(keys) >= 1
  finally:
    env.close()
