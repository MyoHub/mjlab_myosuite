"""Vectorized MyoSuite environment wrapper compatible with mjlab and RSL-RL."""

from __future__ import annotations

import copy
import os
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import vector
from gymnasium.vector import SyncVectorEnv
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .mocks import (
  _MockActionManager,
  _MockCommandManager,
  _MockObservationManager,
  _MockScene,
)
from .sim_compat import create_mock_sim
from .warp_bridge import WarpObservationBridge, is_warp_available

# Set MUJOCO_GL=egl early for headless rendering support
# This must be done BEFORE any MuJoCo imports
if "MUJOCO_GL" not in os.environ:
  os.environ["MUJOCO_GL"] = "egl"


class MyoSuiteVecEnvWrapper(VecEnv, gym.Env):
  """Wraps a vectorized MyoSuite environment to work with mjlab's RslRlVecEnvWrapper.

  This wrapper:
  1. Vectorizes single MyoSuite environments
  2. Converts numpy arrays to torch tensors
  3. Handles observation/action space conversions
  4. Provides a mock cfg attribute for compatibility
  """

  # Mark as vectorized environment for gymnasium
  is_vector_env = True

  # Add metadata for gymnasium
  metadata = {"render_modes": [None, "rgb_array"]}

  def __init__(
    self,
    env: gym.Env | vector.VectorEnv,
    num_envs: int | None = None,
    device: str | torch.device = "cpu",
    clip_actions: float | None = None,
    render_mode: str | None = None,
    physics_backend: Any = None,
  ):
    """Initialize the wrapper.

    Args:
      env: Either a single MyoSuite environment or already vectorized environment
      num_envs: Number of environments (if env is single, will vectorize)
      device: Device to use for tensors
      clip_actions: Optional action clipping value
      render_mode: Render mode for the environment (e.g., "rgb_array" for video recording)
      physics_backend: PhysicsBackend.CPU or WARP (from config); used for data path alignment with mjlab.
    """
    # Initialize gym.Env parent (no-op but required for proper inheritance)
    gym.Env.__init__(self)

    # Store render_mode as an attribute (required for RecordVideo wrapper)
    self._render_mode = render_mode or getattr(env, "render_mode", None)

    # Note: MUJOCO_GL=egl is set at module level for headless rendering support

    # Initialize offline renderer lazily (after scene is created)
    # We'll initialize it in render() or after _setup_manager_compatibility()
    self._offline_renderer = None
    self._offline_renderer_initialized = False

    self.clip_actions = clip_actions

    # Normalize device specification (accepts "CUDA:0", "cuda:0", torch.device, etc.)
    def _normalize_device(d: str | torch.device) -> torch.device:
      if isinstance(d, torch.device):
        return d
      d_str = str(d).strip()
      d_lc = d_str.lower()
      if d_lc.startswith("cuda"):
        # Accept forms like "cuda" or "cuda:0" or uppercase variants
        if ":" in d_lc:
          idx = d_lc.split(":", 1)[1]
          dev = f"cuda:{idx}"
        else:
          dev = "cuda"
        try:
          return torch.device(dev)
        except Exception:  # pragma: no cover - defensive fallback
          return torch.device("cpu")
      # Default to cpu for anything else
      return torch.device("cpu")

    norm_device = _normalize_device(device)
    self.device_str = str(norm_device)
    self.device = norm_device

    # Physics backend (CPU vs WARP) for mjlab-aligned data path; default CPU
    if physics_backend is None:
      try:
        from mjlab_myosuite.config import PhysicsBackend

        self.physics_backend = PhysicsBackend.CPU
      except ImportError:
        self.physics_backend = None
    else:
      self.physics_backend = physics_backend

    # Vectorize if needed
    if isinstance(env, vector.VectorEnv):
      self.env = env
      self.num_envs = env.num_envs
    else:
      # Single environment - need to vectorize
      if num_envs is None:
        num_envs = 1

      # Create environment factories - each lambda captures env in its own closure
      # This ensures each environment is independent (though for MyoSuite they share the same instance)
      def make_env_factory(env_instance: gym.Env):
        """Create a factory that returns the env instance."""
        return lambda: copy.deepcopy(env_instance)

      env_factories = [make_env_factory(env) for _ in range(num_envs)]
      self.env = SyncVectorEnv(env_factories)
      self.num_envs = num_envs

    # Get action/observation spaces
    self.single_action_space = self.env.single_action_space
    original_obs_space = self.env.single_observation_space
    self.action_space = self.env.action_space
    original_obs_space_vec = self.env.observation_space

    # Modify observation space to include both 'policy' and 'critic' groups for RSL-RL
    # RSL-RL expects observations as a dict with 'policy' and 'critic' keys
    if not isinstance(original_obs_space, gym.spaces.Dict):
      # Convert single observation space to Dict with both policy and critic
      # Both use the same space (since we don't have privileged info for critic)
      self.single_observation_space = gym.spaces.Dict(
        {
          "policy": original_obs_space,
          "critic": original_obs_space,
        }
      )
      # Update vectorized observation space
      self.observation_space = gym.vector.utils.batch_space(
        self.single_observation_space, self.num_envs
      )
    elif "policy" not in original_obs_space or "critic" not in original_obs_space:
      # If it's already a Dict but missing policy/critic, add them
      spaces_dict = dict(original_obs_space.spaces)
      if "policy" not in spaces_dict:
        # Use the first available space or create a default
        first_space = (
          next(iter(spaces_dict.values()))
          if spaces_dict
          else gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        )
        spaces_dict["policy"] = first_space
      if "critic" not in spaces_dict:
        # Use policy space for critic
        spaces_dict["critic"] = spaces_dict["policy"]
      self.single_observation_space = gym.spaces.Dict(spaces_dict)
      self.observation_space = gym.vector.utils.batch_space(
        self.single_observation_space, self.num_envs
      )
    else:
      # Already has both policy and critic, use as-is
      self.single_observation_space = original_obs_space
      self.observation_space = original_obs_space_vec

    # Extract action dimension
    if isinstance(self.single_action_space, gym.spaces.Box):
      self.num_actions = int(np.prod(self.single_action_space.shape))
    elif isinstance(self.single_action_space, gym.spaces.Discrete):
      self.num_actions = 1
    else:
      # Try to get shape attribute
      self.num_actions = int(np.prod(getattr(self.single_action_space, "shape", (1,))))

    # Set up mock managers once for ManagerBasedRlEnv/ONNX compatibility.
    self._setup_manager_compatibility()

    # Estimate max episode length (MyoSuite environments typically have timeout)
    # Default to 1000 steps if not available
    unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
    max_episode_length: int | None = None

    # Try to get max_episode_steps from spec
    if hasattr(unwrapped_env, "spec") and unwrapped_env.spec is not None:
      max_episode_length = getattr(unwrapped_env.spec, "max_episode_steps", None)

    # If not found in spec, try direct attribute
    if max_episode_length is None:
      max_episode_length = getattr(unwrapped_env, "max_episode_steps", None)

    # If still None, try to get from the vectorized environment
    if max_episode_length is None and isinstance(self.env, vector.VectorEnv):
      # Try to get from the first environment in the vector
      if hasattr(self.env, "envs") and len(self.env.envs) > 0:
        first_env = self.env.envs[0]
        if hasattr(first_env, "spec") and first_env.spec is not None:
          max_episode_length = getattr(first_env.spec, "max_episode_steps", None)
        if max_episode_length is None:
          max_episode_length = getattr(first_env, "max_episode_steps", None)

    # Set with default fallback
    self.max_episode_length = (
      int(max_episode_length) if max_episode_length is not None else 1000
    )

    # Create mock cfg for compatibility
    self._mock_cfg = self._create_mock_cfg()

    # Track episode lengths
    self.episode_length_buf = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )

    # Track last observation for get_observations()
    self._last_obs_dict: dict[str, Any] = {}

    # Pinned buffer for fast CPU->GPU transfer (fallback when Warp not available)
    self._pinned_policy_buffer: torch.Tensor | None = None
    self._pinned_policy_shape: tuple[int, ...] | None = None

    # Warp bridge: CPU obs → Warp GPU → PyTorch (zero-copy), same as mjlab
    self._warp_bridge: WarpObservationBridge | None = None
    if norm_device.type == "cuda" and is_warp_available():
      try:
        self._warp_bridge = WarpObservationBridge(norm_device)
      except Exception:
        self._warp_bridge = None

    # Create mock sim object for viewer compatibility
    # MyoSuite environments have mj_model and mj_data directly on the unwrapped env
    self._mock_sim = create_mock_sim(self.env)

    # Modify action space if clipping is enabled
    self._modify_action_space()

    # Reset at the start since rsl_rl does not call reset
    obs, _ = self.env.reset()
    self._last_obs_dict = self._convert_obs_to_dict(obs)
    # Verify that observations are on the correct device after initial conversion
    for key, value in self._last_obs_dict.items():
      if isinstance(value, torch.Tensor) and value.device != self.device:
        # Force move to correct device immediately
        self._last_obs_dict[key] = value.to(device=self.device)

  def _create_mock_cfg(self) -> Any:
    """Create a mock cfg for compatibility with RslRlVecEnvWrapper."""
    # Create a dataclass-based mock cfg so it can be converted to dict for logging
    from dataclasses import dataclass, field

    # Lazy import to avoid triggering mjlab import chain
    try:
      from mjlab.viewer.viewer_config import ViewerConfig as MjlabViewerConfig

      ViewerConfigType = MjlabViewerConfig
    except ImportError:
      # Fallback if ViewerConfig is not available
      from dataclasses import dataclass as viewer_dataclass

      @viewer_dataclass
      class ViewerConfigFallback:  # type: ignore[no-redef]
        pass

      ViewerConfigType = ViewerConfigFallback

    @dataclass
    class MockCfg:
      is_finite_horizon: bool = False
      viewer: ViewerConfigType = field(default_factory=ViewerConfigType)  # type: ignore[assignment]

      def to_dict(self) -> dict:
        """Convert to dictionary for wandb logging."""
        from dataclasses import asdict

        return asdict(self)

    return MockCfg()  # type: ignore[return-value]

  @property
  def cfg(self) -> Any:
    """Return mock cfg for compatibility."""
    return self._mock_cfg

  @property
  def sim(self) -> Any:
    """Return mock sim object for viewer compatibility."""
    return self._mock_sim

  @property
  def unwrapped(self) -> "MyoSuiteVecEnvWrapper":
    """Return self as unwrapped (for viewer compatibility).

    This allows the viewer to access env.unwrapped.sim which will
    return our mock sim object with mj_model and mj_data.
    """
    return self

  @property
  def render_mode(self) -> str | None:
    """Get render mode."""
    # Return stored render_mode if set, otherwise get from underlying env
    if hasattr(self, "_render_mode") and self._render_mode is not None:
      return self._render_mode
    return getattr(self.env, "render_mode", None)

  def _initialize_offline_renderer(self) -> None:
    """Initialize offline renderer (called after scene is created)."""
    if self._offline_renderer_initialized:
      return

    # Ensure EGL is set up before initializing renderer
    # (should already be set at module level, but double-check)
    if "MUJOCO_GL" not in os.environ:
      os.environ["MUJOCO_GL"] = "egl"

    try:
      from mjlab.viewer.offscreen_renderer import OffscreenRenderer

      # Get mj_model from the underlying environment
      myosuite_env: Any | None = None
      base_env = self.env
      if base_env is not None and isinstance(base_env, vector.VectorEnv):
        if hasattr(base_env, "envs") and len(base_env.envs) > 0:
          myosuite_env = base_env.envs[0]
          while myosuite_env is not None and hasattr(myosuite_env, "env"):
            next_env = getattr(myosuite_env, "env", None)
            if next_env is None or next_env is myosuite_env:
              break
            myosuite_env = next_env
      elif base_env is not None:
        myosuite_env = base_env

      mj_model = getattr(myosuite_env, "mj_model", getattr(myosuite_env, "model", None))
      if mj_model is not None:
        # Create a minimal viewer config
        try:
          from mjlab.viewer.viewer_config import ViewerConfig

          viewer_cfg = ViewerConfig()
        except ImportError:
          # Fallback: create a simple config dict
          from dataclasses import dataclass

          @dataclass
          class SimpleViewerConfig:
            height: int = 480
            width: int = 640

          viewer_cfg = SimpleViewerConfig()

        # Initialize offline renderer (scene should be created by now)
        # Note: OffscreenRenderer uses mujoco.Renderer internally which requires EGL
        # Type ignore: Mock objects are compatible with viewer interface
        self._offline_renderer = OffscreenRenderer(
          model=mj_model,
          cfg=viewer_cfg,  # type: ignore[arg-type]
          scene=self.scene,  # type: ignore[arg-type]
        )
        self._offline_renderer.initialize()
        self._offline_renderer_initialized = True
    except Exception:  # pragma: no cover - best-effort renderer
      # If offline renderer initialization fails, continue without it
      # render() will return None
      pass

  def render(self) -> np.ndarray | None:
    """Render the environment.

    Uses mjlab's OffscreenRenderer (like ManagerBasedRlEnv does) if available,
    otherwise falls back to the underlying environment's render method.
    """
    # Only render if render_mode is set to rgb_array
    if self.render_mode != "rgb_array":
      return None

    # Initialize offline renderer lazily if not already done
    if not self._offline_renderer_initialized:
      self._initialize_offline_renderer()

    # Use offline renderer if available (like ManagerBasedRlEnv does)
    if self._offline_renderer is not None:
      try:
        # Get mj_data from the underlying environment
        myosuite_env: Any | None = None
        base_env = self.env
        if base_env is not None and isinstance(base_env, vector.VectorEnv):
          if hasattr(base_env, "envs") and len(base_env.envs) > 0:
            myosuite_env = base_env.envs[0]
            while myosuite_env is not None and hasattr(myosuite_env, "env"):
              next_env = getattr(myosuite_env, "env", None)
              if next_env is None or next_env is myosuite_env:
                break
              myosuite_env = next_env
        elif base_env is not None:
          myosuite_env = base_env

        if myosuite_env is not None:
          # OffscreenRenderer.update() expects sim.data format with torch tensors
          # Use our mock sim.data which should provide the right interface
          if hasattr(self, "sim") and self.sim is not None:
            try:
              # Ensure forward kinematics are computed
              import mujoco

              mj_model = getattr(
                myosuite_env,
                "mj_model",
                getattr(myosuite_env, "model", None),
              )
              mj_data = getattr(
                myosuite_env,
                "mj_data",
                getattr(myosuite_env, "data", None),
              )
              if mj_model is not None and mj_data is not None:
                mujoco.mj_forward(mj_model, mj_data)

              # Get sim.data (which should return DataAdapter with torch tensor interface)
              try:
                sim_data = self.sim.data
              except (AttributeError, Exception):
                # If sim.data fails, create DataAdapter directly from mj_data
                import torch

                class DataAdapter:
                  """Adapter to make mj_data look like ManagerBasedRlEnv's sim.data."""

                  def __init__(self, mj_data: Any, mj_model: Any):
                    self._mj_data = mj_data
                    self._mj_model = mj_model
                    # nworld is the number of environments (1 for single env)
                    self.nworld = 1

                  @property
                  def qpos(self):
                    qpos_np = self._mj_data.qpos.copy()
                    return torch.from_numpy(qpos_np).unsqueeze(0)  # Add batch dim

                  @property
                  def qvel(self):
                    qvel_np = self._mj_data.qvel.copy()
                    return torch.from_numpy(qvel_np).unsqueeze(0)  # Add batch dim

                  @property
                  def mocap_pos(self):
                    """Return mocap_pos as torch tensor with batch dimension."""
                    if self._mj_model.nmocap > 0:
                      mocap_pos_np = self._mj_data.mocap_pos.copy()
                      return torch.from_numpy(mocap_pos_np).unsqueeze(0)
                    # Return empty tensor with correct shape
                    return torch.zeros((1, 0, 3), dtype=torch.float32)

                  @property
                  def mocap_quat(self):
                    """Return mocap_quat as torch tensor with batch dimension."""
                    if self._mj_model.nmocap > 0:
                      mocap_quat_np = self._mj_data.mocap_quat.copy()
                      return torch.from_numpy(mocap_quat_np).unsqueeze(0)
                    # Return empty tensor with correct shape
                    return torch.zeros((1, 0, 4), dtype=torch.float32)

                sim_data = DataAdapter(mj_data, mj_model)

              # Update renderer with sim.data (which should have torch tensor interface)
              self._offline_renderer.update(sim_data)
              frame = self._offline_renderer.render()
              if frame is not None:
                return frame
            except Exception:  # pragma: no cover - viewer robustness
              # If update fails, continue to fallback
              pass
      except Exception:  # pragma: no cover - viewer robustness
        # If offline renderer fails, try fallback
        pass

    # Fallback: Try underlying environment's render method
    myosuite_env: Any | None = None
    base_env = self.env
    if base_env is not None and isinstance(base_env, vector.VectorEnv):
      if hasattr(base_env, "envs") and len(base_env.envs) > 0:
        myosuite_env = base_env.envs[0]
        while myosuite_env is not None and hasattr(myosuite_env, "env"):
          next_env = getattr(myosuite_env, "env", None)
          if next_env is None or next_env is myosuite_env:
            break
          myosuite_env = next_env
    elif base_env is not None:
      myosuite_env = base_env

    if myosuite_env is not None and hasattr(myosuite_env, "render"):
      try:
        result = myosuite_env.render()
        if isinstance(result, list) and len(result) > 0:
          frame = result[0] if isinstance(result[0], np.ndarray) else None
          if frame is not None:
            return frame
        elif isinstance(result, np.ndarray):
          return result
      except (NotImplementedError, Exception):
        pass

    # If all else fails, return None
    return None

  @classmethod
  def class_name(cls) -> str:
    """Return class name."""
    return cls.__name__

  # Gymnasium environments expose an attribute `spec: EnvSpec | None`.
  # We keep a plain attribute to satisfy static type checkers.
  spec: Any = None

  def seed(self, seed: int = -1) -> int:
    """Set seed for environment."""
    if hasattr(self.env, "seed"):
      return self.env.seed(seed)  # type: ignore[attr-defined]
    return seed

  def get_observations(self) -> TensorDict:
    """Get observations as TensorDict.

    CRITICAL: RSL-RL expects observations on the same device as the policy.
    This method ensures all observation tensors are on self.device.
    """
    # If _last_obs_dict is empty, we need to reset the environment first
    # This can happen if get_observations() is called before the first step
    if not self._last_obs_dict:
      obs, _ = self.env.reset()
      self._last_obs_dict = self._convert_obs_to_dict(obs)
      # Verify that observations are on the correct device after initial conversion
      for key, value in self._last_obs_dict.items():
        if isinstance(value, torch.Tensor) and value.device != self.device:
          # Force move to correct device immediately
          self._last_obs_dict[key] = value.to(device=self.device)

    # Rebuild observation dict, ensuring all tensors are on the correct device
    # This is necessary because tensors might have been created on CPU initially
    obs_dict_on_device: dict[str, torch.Tensor] = {}
    for key, value in self._last_obs_dict.items():
      if isinstance(value, torch.Tensor):
        # CRITICAL: Always create a NEW tensor on the target device
        # This ensures the tensor is definitely on the correct device
        # Use .contiguous() to ensure it's a proper tensor, not a view
        if value.device != self.device:
          # Move to device - this creates a new tensor if needed
          obs_dict_on_device[key] = value.to(device=self.device).contiguous()
        else:
          # Even if on correct device, ensure it's contiguous
          obs_dict_on_device[key] = value.contiguous()
      else:
        # For non-tensors, convert to tensor on target device
        if isinstance(value, np.ndarray):
          obs_dict_on_device[key] = (
            torch.from_numpy(value)
            .to(device=self.device, dtype=torch.float32)
            .contiguous()
          )
        else:
          obs_dict_on_device[key] = torch.tensor(
            value, device=self.device, dtype=torch.float32
          ).contiguous()

    # Create TensorDict
    td = TensorDict(obs_dict_on_device, batch_size=[self.num_envs])

    # RSL-RL expects policy/critic on correct device; avoid redundant clone when already on device
    non_blocking = self.device.type == "cuda"
    if "policy" in td:
      policy_val = td["policy"]
      if isinstance(policy_val, torch.Tensor):
        if policy_val.device == self.device:
          td["policy"] = policy_val.contiguous()
        else:
          td["policy"] = policy_val.to(
            device=self.device, non_blocking=non_blocking
          ).contiguous()
      if "critic" in td:
        critic_val = td["critic"]
        if isinstance(critic_val, torch.Tensor):
          if critic_val.device == self.device:
            td["critic"] = critic_val.contiguous()
          else:
            td["critic"] = critic_val.to(
              device=self.device, non_blocking=non_blocking
            ).contiguous()

    return td

  def reset(
    self,
    *,
    seed: int | None = None,
    options: dict | None = None,
  ) -> tuple[TensorDict, dict]:
    """Reset the environment (gym-compatible signature)."""
    try:
      obs, info = self.env.reset(seed=seed, options=options)
    except TypeError:
      # Fallback for older envs without seed/options
      obs, info = self.env.reset()

    # Update forward kinematics for visualization after reset
    # This is critical for the viewer to show the initial state
    if hasattr(self, "_mock_sim") and hasattr(self._mock_sim, "_env"):
      import mujoco

      # Support both standard and mjx/warp versions
      env = self._mock_sim._env
      mj_model = getattr(env, "mj_model", getattr(env, "model", None))
      mj_data = getattr(env, "mj_data", getattr(env, "data", None))
      if mj_model is not None and mj_data is not None:
        mujoco.mj_forward(mj_model, mj_data)

    # Convert to torch tensors and store for get_observations()
    obs_dict = self._convert_obs_to_dict(obs)
    # CRITICAL: Ensure all tensors in obs_dict are on the correct device
    for key, value in obs_dict.items():
      if isinstance(value, torch.Tensor) and value.device != self.device:
        obs_dict[key] = value.to(device=self.device)
    self._last_obs_dict = obs_dict
    self.episode_length_buf.zero_()

    return TensorDict(obs_dict, batch_size=[self.num_envs]), info

  def step(  # type: ignore[override]
    self, actions: torch.Tensor | np.ndarray
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    """Step the environment.

    Returns 4 values for rsl_rl compatibility: (obs, rewards, dones, extras)
    where dones = terminated | truncated.
    """
    # Convert actions to numpy (hot path: avoid sync when possible)
    if isinstance(actions, torch.Tensor):
      if actions.is_cuda:
        actions_np = actions.cpu().numpy()
      else:
        actions_np = actions.numpy()
    else:
      actions_np = np.asarray(actions, dtype=np.float32)

    # Clip actions if needed
    if self.clip_actions is not None:
      actions_np = np.clip(actions_np, -self.clip_actions, self.clip_actions)

    # Step environment
    step_result = self.env.step(actions_np)

    # Handle both old and new Gym API
    if len(step_result) == 4:
      # old API: obs, reward, done, info
      obs, rew, done, info = step_result  # type: ignore[misc]
      terminated = done
      # Convert False to tensor with same shape as terminated
      if isinstance(terminated, (bool, np.bool_)):
        truncated = np.array(False, dtype=bool)
      elif isinstance(terminated, np.ndarray):
        truncated = np.zeros_like(terminated, dtype=bool)
      else:
        truncated = False
    elif len(step_result) == 5:
      # new API: obs, reward, terminated, truncated, info
      obs, rew, terminated, truncated, info = step_result  # type: ignore[misc]
      # No need to compute done - we have terminated and truncated separately
    else:
      raise ValueError(
        f"Unexpected number of values returned from env.step: {len(step_result)}"
      )

    # Update forward kinematics only when rendering is needed (skip for training hot path)
    if (
      self._render_mode is not None
      and hasattr(self, "_mock_sim")
      and hasattr(self._mock_sim, "_env")
    ):
      import mujoco

      env = self._mock_sim._env
      mj_model = getattr(env, "mj_model", getattr(env, "model", None))
      mj_data = getattr(env, "mj_data", getattr(env, "data", None))
      if mj_model is not None and mj_data is not None:
        mujoco.mj_forward(mj_model, mj_data)

    # Convert to torch tensors and store for get_observations()
    obs_dict = self._convert_obs_to_dict(obs)
    # CRITICAL: Ensure all tensors in obs_dict are on the correct device
    for key, value in obs_dict.items():
      if isinstance(value, torch.Tensor) and value.device != self.device:
        obs_dict[key] = value.to(device=self.device)
    self._last_obs_dict = obs_dict
    rew_tensor = torch.as_tensor(rew, device=self.device, dtype=torch.float32)
    terminated_tensor = torch.as_tensor(
      terminated, device=self.device, dtype=torch.bool
    )
    truncated_tensor = torch.as_tensor(truncated, device=self.device, dtype=torch.bool)

    # Update episode lengths
    self.episode_length_buf += 1

    # Reset episode lengths for terminated/truncated envs
    done_mask = terminated_tensor | truncated_tensor
    self.episode_length_buf[done_mask] = 0

    # Combine terminated and truncated into dones for rsl_rl compatibility
    dones_tensor = done_mask

    # Add time_outs and other info to extras
    extras = info.copy() if isinstance(info, dict) else {}
    extras["time_outs"] = truncated_tensor
    extras["terminated"] = terminated_tensor
    extras["truncated"] = truncated_tensor

    # Return 4 values for rsl_rl compatibility: (obs, rewards, dones, extras)
    return (
      TensorDict(obs_dict, batch_size=[self.num_envs]),
      rew_tensor,
      dones_tensor,
      extras,
    )

  def _numpy_to_device(self, arr: np.ndarray, key: str = "policy") -> torch.Tensor:
    """Copy numpy array to device. On CUDA uses Warp bridge (zero-copy) when available, else pinned transfer."""
    arr = np.asarray(arr, dtype=np.float32)
    if self.device.type != "cuda":
      return torch.from_numpy(arr).to(device=self.device, dtype=torch.float32)
    if self._warp_bridge is not None:
      return self._warp_bridge.numpy_to_torch(key, arr)
    shape = arr.shape
    if self._pinned_policy_buffer is None or self._pinned_policy_buffer.shape != shape:
      self._pinned_policy_buffer = torch.empty(
        shape, dtype=torch.float32, pin_memory=True
      )
      self._pinned_policy_shape = shape
    self._pinned_policy_buffer.copy_(torch.from_numpy(arr))
    return self._pinned_policy_buffer.to(
      device=self.device, dtype=torch.float32, non_blocking=True
    )

  def _convert_obs_to_dict(self, obs: Any) -> dict[str, Any]:
    """Convert observations to dictionary of torch tensors.

    RSL-RL expects observation groups like 'policy' and 'critic'.
    For MyoSuite, we map all observations to both 'policy' and 'critic' groups
    (since we don't have privileged information for critic).
    """
    if isinstance(obs, dict):
      obs_dict: dict[str, Any] = {}
      policy_arrays: list[np.ndarray] = []
      policy_tensors: list[torch.Tensor] = []
      for key, value in obs.items():
        if key in ["policy", "critic"]:
          if isinstance(value, torch.Tensor):
            obs_dict[key] = value.to(device=self.device, dtype=torch.float32)
          elif isinstance(value, np.ndarray):
            obs_dict[key] = self._numpy_to_device(value, key=key)
          else:
            obs_dict[key] = torch.tensor(value, device=self.device, dtype=torch.float32)
          continue
        if isinstance(value, torch.Tensor):
          policy_tensors.append(value.to(device=self.device, dtype=torch.float32))
        elif isinstance(value, np.ndarray):
          policy_arrays.append(np.asarray(value, dtype=np.float32))
        else:
          policy_tensors.append(
            torch.tensor(value, device=self.device, dtype=torch.float32)
          )
      if policy_arrays:
        policy_obs = self._numpy_to_device(
          np.concatenate(policy_arrays, axis=-1)
          if len(policy_arrays) > 1
          else policy_arrays[0],
          key="policy",
        )
        obs_dict["policy"] = policy_obs
        if "critic" not in obs_dict:
          obs_dict["critic"] = policy_obs
      elif policy_tensors:
        policy_obs = (
          torch.cat(policy_tensors, dim=-1)
          if len(policy_tensors) > 1
          else policy_tensors[0]
        )
        obs_dict["policy"] = policy_obs
        if "critic" not in obs_dict:
          obs_dict["critic"] = policy_obs
      return obs_dict
    if isinstance(obs, np.ndarray):
      # Single observation array - Warp bridge or pinned buffer on GPU
      policy_obs = self._numpy_to_device(obs, key="policy")
      return {"policy": policy_obs, "critic": policy_obs}
    if isinstance(obs, (list, tuple)):
      # Handle list/tuple of observations
      if isinstance(obs[0], dict):
        # List of dicts - convert to dict of arrays
        obs_dict = {}
        for key in obs[0].keys():
          tensor = torch.as_tensor(
            np.array([o[key] for o in obs]),
            device=self.device,
            dtype=torch.float32,
          )
          if key in ["policy", "critic"]:
            obs_dict[key] = tensor
          else:
            obs_dict["policy"] = tensor
        # Ensure 'critic' exists (use 'policy' if not provided)
        if "critic" not in obs_dict and "policy" in obs_dict:
          obs_dict["critic"] = obs_dict["policy"]
        return obs_dict
      # List of arrays - map to both 'policy' and 'critic' groups
      policy_obs = torch.from_numpy(np.array(obs)).to(
        device=self.device, dtype=torch.float32
      )
      return {"policy": policy_obs, "critic": policy_obs}
    # Fallback
    if isinstance(obs, np.ndarray):
      policy_obs = self._numpy_to_device(obs, key="policy")
    else:
      policy_obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
    return {"policy": policy_obs, "critic": policy_obs}

  def _setup_manager_compatibility(self) -> None:
    """Set up mock managers for ManagerBasedRlEnv compatibility.

    This allows the environment to work with ONNX export utilities that expect
    ManagerBasedRlEnv structure (scene, action_manager, observation_manager, etc.).
    """
    # Create mock scene
    self.scene = _MockScene(self.num_envs)

    # Create mock action manager
    self.action_manager = _MockActionManager(self.single_action_space, self.num_envs)

    # Create mock observation manager
    self.observation_manager = _MockObservationManager(self.single_observation_space)

    # Create mock command manager
    self.command_manager = _MockCommandManager()

  def close(self) -> None:
    """Close the environment."""
    if self._warp_bridge is not None:
      self._warp_bridge.clear_buffers()
    return self.env.close()

  def _modify_action_space(self) -> None:
    """Modify action space if clipping is enabled."""
    if self.clip_actions is None:
      return

    if isinstance(self.single_action_space, gym.spaces.Box):
      self.single_action_space = gym.spaces.Box(
        low=-self.clip_actions,
        high=self.clip_actions,
        shape=(self.num_actions,),
      )
      self.action_space = gym.vector.utils.batch_space(
        self.single_action_space, self.num_envs
      )
