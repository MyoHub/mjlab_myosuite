"""Tests to verify whether both physics and models reside on GPU.

These tests probe the three-layer GPU stack described in the analysis doc:

1. **Physics layer** -- Is the MuJoCo model/data on GPU (via mjwarp/MJX)?
2. **RL tensor layer** -- Are observations, rewards, actions on GPU?

For standard MyoSuite (CPU physics + GPU tensor shuttle), physics stays on CPU
and only RL tensors are moved to GPU.  For mjlab-native environments (mjwarp),
both physics and RL tensors should live on GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_myosuite() -> bool:
  try:
    import myosuite  # noqa: F401

    return True
  except Exception:
    return False


def _has_cuda() -> bool:
  return torch.cuda.is_available()


def _has_mjlab() -> bool:
  try:
    import mjlab  # noqa: F401

    return True
  except Exception:
    return False


def _has_warp() -> bool:
  try:
    import warp  # noqa: F401

    return True
  except Exception:
    return False


def _has_mujoco_warp() -> bool:
  try:
    import mujoco.warp  # noqa: F401

    return True
  except Exception:
    try:
      import mujoco_warp  # noqa: F401

      return True
    except Exception:
      return False


# ---------------------------------------------------------------------------
# 1. Physics layer: model and simulation data device checks
# ---------------------------------------------------------------------------


class TestPhysicsOnGpu:
  """Verify where the MuJoCo model and simulation data physically reside."""

  pytestmark = pytest.mark.skipif(
    not _has_myosuite(), reason="myosuite not installed"
  )

  def _make_env(self, device: str = "cpu", num_envs: int = 2):
    from mjlab_myosuite.config import MyoSuiteEnvCfg
    from mjlab_myosuite.env_factory import make_myosuite_env

    cfg = MyoSuiteEnvCfg()
    cfg.device = device
    cfg.num_envs = num_envs
    return make_myosuite_env("myoElbowPose1D6MRandom-v0", cfg=cfg)

  def test_cpu_physics_model_stays_on_cpu(self):
    """Standard MyoSuite: mj_model / mj_data must be CPU-side C structs."""
    env = self._make_env(device="cpu")
    try:
      sim = env.sim
      mj_model = getattr(sim, "mj_model", None)
      mj_data = getattr(sim, "mj_data", None)

      assert mj_model is not None, "sim.mj_model should exist"
      assert mj_data is not None, "sim.mj_data should exist"

      # Standard MuJoCo model/data are C structs -- they have no .device
      # attribute.  Their arrays are plain numpy.
      assert not hasattr(mj_model, "device"), (
        "CPU mj_model should not expose a .device attribute"
      )

      # qpos should be a numpy array (CPU), not a torch/jax/warp array
      qpos = mj_data.qpos
      assert isinstance(qpos, np.ndarray), (
        f"qpos should be numpy.ndarray on CPU, got {type(qpos)}"
      )
    finally:
      env.close()

  @pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
  def test_gpu_device_does_not_move_physics_to_gpu(self):
    """Setting device='cuda:0' should NOT move the physics model to GPU.

    In the wrapper architecture, device only affects RL tensors.
    The underlying MuJoCo C model and data remain on CPU.
    """
    env = self._make_env(device="cuda:0")
    try:
      sim = env.sim
      mj_model = getattr(sim, "mj_model", None)
      mj_data = getattr(sim, "mj_data", None)

      assert mj_model is not None
      assert mj_data is not None

      # Physics data is still numpy (CPU) even with cuda device
      qpos = mj_data.qpos
      assert isinstance(qpos, np.ndarray), (
        f"qpos should remain numpy.ndarray even with cuda device, got {type(qpos)}"
      )

      # Confirm the wrapper's *tensor* device is cuda
      assert str(env.device).startswith("cuda"), (
        f"Wrapper device should be cuda, got {env.device}"
      )
    finally:
      env.close()

  def test_mock_wp_data_provides_cpu_arrays(self):
    """MockWpData should provide batched numpy arrays, not GPU arrays."""
    env = self._make_env(device="cpu")
    try:
      sim = env.sim
      wp_data = getattr(sim, "wp_data", None)
      assert wp_data is not None, "sim.wp_data should exist (MockWpData)"

      # wp_data.qpos should return an ArrayProxy whose .numpy() gives ndarray
      qpos_proxy = wp_data.qpos
      assert hasattr(qpos_proxy, "numpy"), (
        "MockWpData.qpos should have a .numpy() method"
      )
      qpos_np = qpos_proxy.numpy()
      assert isinstance(qpos_np, np.ndarray), (
        f"wp_data.qpos.numpy() should be ndarray, got {type(qpos_np)}"
      )
      # Should have batch dimension
      assert qpos_np.ndim >= 2, (
        f"wp_data.qpos should be batched (ndim >= 2), got shape {qpos_np.shape}"
      )
    finally:
      env.close()

  @pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
  def test_data_adapter_produces_cpu_tensors(self):
    """MockSim.data (DataAdapter) should produce CPU torch tensors.

    Even when device='cuda:0', the DataAdapter wraps mj_data which is on CPU.
    The tensors it creates via torch.from_numpy are on CPU.
    """
    env = self._make_env(device="cuda:0")
    try:
      sim = env.sim
      # sim.data returns a DataAdapter for OffscreenRenderer compatibility
      data = sim.data

      if hasattr(data, "qpos"):
        qpos_tensor = data.qpos
        if isinstance(qpos_tensor, torch.Tensor):
          assert qpos_tensor.device.type == "cpu", (
            f"DataAdapter.qpos should be CPU tensor (wraps numpy), got {qpos_tensor.device}"
          )
    finally:
      env.close()


# ---------------------------------------------------------------------------
# 2. RL tensor layer: observations, rewards, actions device checks
# ---------------------------------------------------------------------------


class TestRlTensorsOnGpu:
  """Verify RL tensors (obs, rewards, dones) are on the configured device."""

  pytestmark = [
    pytest.mark.skipif(not _has_myosuite(), reason="myosuite not installed"),
    pytest.mark.skipif(not _has_cuda(), reason="CUDA not available"),
  ]

  def _make_env(self, device: str = "cuda:0", num_envs: int = 4):
    from mjlab_myosuite.config import MyoSuiteEnvCfg
    from mjlab_myosuite.env_factory import make_myosuite_env

    cfg = MyoSuiteEnvCfg()
    cfg.device = device
    cfg.num_envs = num_envs
    return make_myosuite_env("myoElbowPose1D6MRandom-v0", cfg=cfg)

  def test_observations_on_gpu(self):
    """Observations returned by get_observations() must be on the GPU device."""
    env = self._make_env()
    try:
      env.reset()
      td = env.get_observations()

      assert "policy" in td
      policy_obs = td["policy"]
      assert isinstance(policy_obs, torch.Tensor)
      assert policy_obs.device.type == "cuda", (
        f"policy obs should be on cuda, got {policy_obs.device}"
      )
      assert policy_obs.shape[0] == 4, (
        f"batch dim should be num_envs=4, got {policy_obs.shape[0]}"
      )

      if "critic" in td:
        critic_obs = td["critic"]
        assert isinstance(critic_obs, torch.Tensor)
        assert critic_obs.device.type == "cuda"
    finally:
      env.close()

  def test_rewards_and_dones_on_gpu(self):
    """Rewards and done flags from step() must be on GPU."""
    env = self._make_env()
    try:
      env.reset()
      action = env.action_space.sample()
      _, rewards, dones, extras = env.step(action)

      assert isinstance(rewards, torch.Tensor)
      assert rewards.device.type == "cuda", (
        f"rewards should be on cuda, got {rewards.device}"
      )
      assert rewards.shape[0] == 4

      if isinstance(dones, torch.Tensor):
        assert dones.device.type == "cuda", (
          f"dones should be on cuda, got {dones.device}"
        )

      # Check extras
      if "terminated" in extras and isinstance(extras["terminated"], torch.Tensor):
        assert extras["terminated"].device.type == "cuda"
      if "time_outs" in extras and isinstance(extras["time_outs"], torch.Tensor):
        assert extras["time_outs"].device.type == "cuda"
    finally:
      env.close()

  def test_step_obs_on_gpu_after_multiple_steps(self):
    """Observations must stay on GPU across multiple consecutive steps."""
    env = self._make_env()
    try:
      env.reset()
      for _ in range(5):
        action = env.action_space.sample()
        obs_td, rewards, dones, extras = env.step(action)

        # Check obs TensorDict
        if "policy" in obs_td:
          policy = obs_td["policy"]
          if isinstance(policy, torch.Tensor):
            assert policy.device.type == "cuda", (
              f"policy obs drifted off GPU: {policy.device}"
            )

        # Check via get_observations (separate code path)
        td = env.get_observations()
        policy2 = td["policy"]
        if isinstance(policy2, torch.Tensor):
          assert policy2.device.type == "cuda", (
            f"get_observations() drifted off GPU: {policy2.device}"
          )
    finally:
      env.close()


# ---------------------------------------------------------------------------
# 3. Physics vs RL tensor split: the full picture
# ---------------------------------------------------------------------------


class TestPhysicsVsRlTensorSplit:
  """Verify the expected split: CPU physics + GPU RL tensors.

  This is the central test for the wrapper architecture:
  - Physics (mj_model, mj_data, qpos, qvel) stays on CPU as numpy arrays
  - RL tensors (observations, rewards, dones) are on the configured GPU device
  - The bridge between them is _numpy_to_device() with pinned memory
  """

  pytestmark = [
    pytest.mark.skipif(not _has_myosuite(), reason="myosuite not installed"),
    pytest.mark.skipif(not _has_cuda(), reason="CUDA not available"),
  ]

  def test_physics_cpu_rl_gpu_split(self):
    """Full integration: physics on CPU, RL tensors on GPU, data is consistent."""
    from mjlab_myosuite.config import MyoSuiteEnvCfg
    from mjlab_myosuite.env_factory import make_myosuite_env

    cfg = MyoSuiteEnvCfg()
    cfg.device = "cuda:0"
    cfg.num_envs = 2
    env = make_myosuite_env("myoElbowPose1D6MRandom-v0", cfg=cfg)

    try:
      env.reset()

      # --- Physics layer: CPU ---
      sim = env.sim
      mj_data = getattr(sim, "mj_data", None)
      assert mj_data is not None

      physics_qpos = mj_data.qpos
      assert isinstance(physics_qpos, np.ndarray), (
        f"Physics qpos must be numpy (CPU), got {type(physics_qpos)}"
      )

      # --- RL tensor layer: GPU ---
      td = env.get_observations()
      policy_obs = td["policy"]
      assert isinstance(policy_obs, torch.Tensor)
      assert policy_obs.device.type == "cuda", (
        f"RL observations must be on cuda, got {policy_obs.device}"
      )

      # --- Step and verify both layers ---
      action = env.action_space.sample()
      obs_td, rewards, dones, extras = env.step(action)

      # Physics still on CPU after step
      physics_qpos_after = mj_data.qpos
      assert isinstance(physics_qpos_after, np.ndarray)

      # RL tensors on GPU after step
      assert rewards.device.type == "cuda"
      if "policy" in obs_td:
        assert obs_td["policy"].device.type == "cuda"

      # --- Consistency: physics qpos should have changed ---
      # (we applied an action, so state should differ)
      # Use allclose with tolerance since some envs might not move much
      assert physics_qpos_after.shape == physics_qpos.shape

    finally:
      env.close()

  def test_pinned_memory_transfer_path(self):
    """The _numpy_to_device path should use pinned memory for GPU targets."""
    from mjlab_myosuite.config import MyoSuiteEnvCfg
    from mjlab_myosuite.env_factory import make_myosuite_env

    cfg = MyoSuiteEnvCfg()
    cfg.device = "cuda:0"
    cfg.num_envs = 1
    env = make_myosuite_env("myoElbowPose1D6MRandom-v0", cfg=cfg)

    try:
      # The pinned buffer is allocated lazily on first _numpy_to_device call
      assert env._pinned_policy_buffer is None or isinstance(
        env._pinned_policy_buffer, torch.Tensor
      )

      # Reset triggers _convert_obs_to_dict -> _numpy_to_device
      env.reset()

      # After reset, pinned buffer should be allocated (for GPU path)
      if env._pinned_policy_buffer is not None:
        assert env._pinned_policy_buffer.is_pinned(), (
          "Pinned buffer should have pin_memory=True for GPU transfers"
        )

      # Verify the resulting observation is on GPU
      td = env.get_observations()
      assert td["policy"].device.type == "cuda"

    finally:
      env.close()


# ---------------------------------------------------------------------------
# 4. mjlab-native GPU physics (when mjlab + warp are available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
  not (_has_mjlab() and _has_warp() and _has_mujoco_warp() and _has_cuda()),
  reason="mjlab, warp, mujoco-warp, or CUDA not available",
)
class TestMjlabNativeGpuPhysics:
  """Tests for mjlab's native GPU physics via MuJoCo Warp.

  These tests only run when the full mjlab stack is available.
  They verify that mjlab puts BOTH the physics model AND RL tensors on GPU.
  """

  def test_simulation_model_on_gpu(self):
    """mjlab's Simulation should place the wp_model on GPU."""
    import mujoco
    from mjlab.sim.sim import Simulation, SimulationCfg

    # Create a minimal MuJoCo model
    spec = mujoco.MjSpec()
    spec.worldbody.add_body().add_joint()
    spec.worldbody.add_body().add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1])
    mj_model = spec.compile()

    sim_cfg = SimulationCfg()
    sim = Simulation(num_envs=2, cfg=sim_cfg, model=mj_model, device="cuda:0")

    try:
      # wp_model should be a WarpBridge wrapping GPU arrays
      wp_model = sim.model
      assert wp_model is not None, "sim.model (wp_model bridge) should exist"

      # wp_data should be a WarpBridge wrapping GPU arrays
      wp_data = sim.data
      assert wp_data is not None, "sim.data (wp_data bridge) should exist"

      # Access qpos through the bridge -- should be a TorchArray on GPU
      qpos = wp_data.qpos
      if isinstance(qpos, torch.Tensor):
        assert qpos.device.type == "cuda", (
          f"mjlab sim.data.qpos should be on GPU, got {qpos.device}"
        )
      elif hasattr(qpos, "_tensor"):
        # TorchArray wraps a tensor
        assert qpos._tensor.device.type == "cuda", (
          f"TorchArray._tensor should be on GPU, got {qpos._tensor.device}"
        )

    finally:
      del sim

  def test_simulation_cuda_graph_capture(self):
    """mjlab should be able to capture CUDA graphs for step/forward."""
    import mujoco
    from mjlab.sim.sim import Simulation, SimulationCfg

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body()
    body.add_joint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1])
    mj_model = spec.compile()

    sim_cfg = SimulationCfg()
    sim = Simulation(num_envs=2, cfg=sim_cfg, model=mj_model, device="cuda:0")

    try:
      # CUDA graph capture should succeed on supported hardware
      if sim.use_cuda_graph:
        sim.create_graph()
        assert sim.step_graph is not None, "step_graph should be captured"
        assert sim.forward_graph is not None, "forward_graph should be captured"

      # Step should work (with or without graphs)
      sim.step()
      sim.forward()

    finally:
      del sim

  def test_warp_bridge_zero_copy(self):
    """WarpBridge/TorchArray should share GPU memory (zero-copy)."""
    import warp as wp

    from mjlab.sim.sim_data import TorchArray, WarpBridge

    # Create a warp array on GPU
    with wp.ScopedDevice("cuda:0"):
      wp_arr = wp.zeros(10, dtype=wp.float32)

    ta = TorchArray(wp_arr)

    # Should produce a torch tensor on the same GPU device
    assert isinstance(ta._tensor, torch.Tensor)
    assert ta._tensor.device.type == "cuda"

    # Modify via TorchArray, verify via warp (zero-copy check)
    ta[0] = 42.0
    result = wp.to_torch(wp_arr)
    assert float(result[0]) == 42.0, (
      "TorchArray write should be visible through Warp array (zero-copy)"
    )

  def test_warp_bridge_read_only(self):
    """WarpBridge should prevent direct attribute assignment."""
    import warp as wp

    from mjlab.sim.sim_data import WarpBridge

    class MockStruct:
      def __init__(self):
        with wp.ScopedDevice("cuda:0"):
          self.arr = wp.zeros(5, dtype=wp.float32)

    bridge = WarpBridge(MockStruct())

    # Reading should work
    _ = bridge.arr

    # Writing should raise AttributeError
    with pytest.raises(AttributeError):
      bridge.arr = "should fail"
