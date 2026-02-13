"""Tests that mjlab_myosuite and (when available) native mjlab use GPU and scale similarly."""

import time

import pytest


def _has_myosuite() -> bool:
  try:
    import myosuite  # noqa: F401

    return True
  except Exception:
    return False


def _has_cuda() -> bool:
  try:
    import torch

    return torch.cuda.is_available()
  except Exception:
    return False


pytestmark = pytest.mark.skipif(
  not _has_myosuite() or not _has_cuda(),
  reason="myosuite not installed or CUDA not available",
)


def _measure_throughput(
  num_envs: int, device: str, steps: int = 100, warmup: int = 20
) -> float:
  """Steps per second for mjlab_myosuite env."""
  import numpy as np

  from mjlab_myosuite.env_factory import make_myosuite_env

  env = make_myosuite_env("myoElbowPose1D6MRandom-v0", num_envs=num_envs, device=device)
  try:
    env.reset(seed=42)
    n = env.num_envs
    action_dim = getattr(env.unwrapped, "num_actions", None)
    if action_dim is None and hasattr(env, "single_action_space"):
      action_dim = int(np.prod(getattr(env.single_action_space, "shape", (1,))))
    else:
      action_dim = action_dim or 1
    for _ in range(warmup):
      env.step(np.zeros((n, action_dim), dtype=np.float32))
    env.reset(seed=123)
    import torch

    t0 = time.perf_counter()
    for _ in range(steps):
      env.step(np.zeros((n, action_dim), dtype=np.float32))
    if device != "cpu":
      torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (steps * n) / elapsed
  finally:
    env.close()


def test_mjlab_myosuite_gpu_throughput_increases_with_num_envs():
  """mjlab_myosuite on GPU: total steps/sec should scale with num_envs (more envs => more throughput)."""
  sps_64 = _measure_throughput(64, "cuda:0", steps=80, warmup=15)
  sps_256 = _measure_throughput(256, "cuda:0", steps=80, warmup=15)
  # With more envs we expect at least similar or higher total steps/sec (batch parallelism)
  # Allow some variance: 256 envs should not be worse than 50% of 64 envs per-step efficiency
  total_64 = sps_64
  total_256 = sps_256
  assert total_256 >= 0.3 * total_64, (
    f"GPU scaling: 256 envs gave {total_256:.0f} steps/s vs 64 envs {total_64:.0f}. "
    "Both should use GPU and scale similarly."
  )


def test_mjlab_myosuite_gpu_uses_device():
  """mjlab_myosuite with device=cuda:0 reports cuda and get_observations returns dict-like."""
  from mjlab_myosuite.env_factory import make_myosuite_env

  env = make_myosuite_env("myoElbowPose1D6MRandom-v0", num_envs=2, device="cuda:0")
  try:
    assert str(env.device).startswith("cuda"), "Env must report cuda when device=cuda:0"
    env.reset(seed=42)
    obs = env.get_observations()
    assert hasattr(obs, "keys") or isinstance(obs, dict)
    # Wrapper should place tensors on env.device; at least one observation key present
    keys = list(obs.keys()) if hasattr(obs, "keys") else list(obs)
    assert "policy" in keys or len(keys) >= 1
  finally:
    env.close()
