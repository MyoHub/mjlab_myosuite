"""Wrapper for MyoSuite environments to work with mjlab's training infrastructure."""

import copy
import os
from typing import Any

# Set MUJOCO_GL=egl early for headless rendering support
# This must be done BEFORE any MuJoCo imports
if "MUJOCO_GL" not in os.environ:
  os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
import numpy as np
import torch
from gymnasium import vector
from gymnasium.vector import SyncVectorEnv
from rsl_rl.env import VecEnv
from tensordict import TensorDict

# ManagerBasedRlEnvCfg is only used for type hints, imported lazily when needed


class _MockActionManager:
  """Mock action manager for ManagerBasedRlEnv compatibility."""

  def __init__(self, action_space, num_envs: int):
    self.action_space = action_space
    self.num_envs = num_envs
    self.active_terms = ["joint_pos"]  # Default action term name

  def get_term(self, name: str):
    """Get action term by name."""

    # Return a mock action term object
    class _MockActionTerm:
      def __init__(self, action_space):
        self._scale = self._get_action_scale(action_space)

      def _get_action_scale(self, action_space):
        """Get action scale from action space."""
        if hasattr(action_space, "low") and hasattr(action_space, "high"):
          low = np.array(action_space.low)
          high = np.array(action_space.high)
          # Return scale as (high - low) / 2
          scale = (high - low) / 2.0
          return torch.tensor(scale, dtype=torch.float32)
        return torch.ones(self._get_action_dim(action_space), dtype=torch.float32)

      def _get_action_dim(self, action_space):
        """Get action dimension."""
        if hasattr(action_space, "shape"):
          return int(np.prod(action_space.shape))
        return 1

    return _MockActionTerm(self.action_space)


class _MockObservationManager:
  """Mock observation manager for ManagerBasedRlEnv compatibility."""

  def __init__(self, observation_space):
    self.observation_space = observation_space
    self.active_terms = {"policy": self._get_observation_names(observation_space)}

  def _get_observation_names(self, observation_space):
    """Get observation names from observation space."""
    if isinstance(observation_space, gym.spaces.Dict):
      if "policy" in observation_space.spaces:
        policy_space = observation_space.spaces["policy"]
        if hasattr(policy_space, "shape") and policy_space.shape is not None:
          dim = int(np.prod(policy_space.shape))
          return [f"obs_{i}" for i in range(dim)]
      # Fallback: use first space
      if observation_space.spaces:
        first_space = next(iter(observation_space.spaces.values()))
        if hasattr(first_space, "shape") and first_space.shape is not None:
          dim = int(np.prod(first_space.shape))
          return [f"obs_{i}" for i in range(dim)]
    elif hasattr(observation_space, "shape") and observation_space.shape is not None:
      dim = int(np.prod(observation_space.shape))
      return [f"obs_{i}" for i in range(dim)]
    return ["obs_0"]


class _MockCommandManager:
  """Mock command manager for ManagerBasedRlEnv compatibility."""

  def __init__(self):
    self.active_terms = []  # MyoSuite doesn't use commands by default


class _MockScene:
  """Mock scene for ManagerBasedRlEnv compatibility."""

  def __init__(self, num_envs: int):
    self.num_envs = num_envs

  def __getitem__(self, key: str):
    """Get entity by name (returns mock robot entity)."""

    # Return a mock robot entity
    class _MockRobot:
      def __init__(self):
        self.joint_names = []  # Will be populated if needed
        self.spec = _MockSpec()
        self.data = _MockData()

    class _MockSpec:
      def __init__(self):
        self.actuators = []  # Empty actuators list

    class _MockData:
      def __init__(self):
        self.default_joint_pos = torch.zeros(1, 0)  # Empty default positions

    return _MockRobot()


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
  ):
    """Initialize the wrapper.

    Args:
      env: Either a single MyoSuite environment or already vectorized environment
      num_envs: Number of environments (if env is single, will vectorize)
      device: Device to use for tensors
      clip_actions: Optional action clipping value
      render_mode: Render mode for the environment (e.g., "rgb_array" for video recording)
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
        except Exception:
          return torch.device("cpu")
      # Default to cpu for anything else
      return torch.device("cpu")

    norm_device = _normalize_device(device)
    self.device_str = str(norm_device)
    self.device = norm_device

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
      def make_env_factory(env_instance):
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

    # Create mock managers for ManagerBasedRlEnv compatibility (needed for ONNX export)
    self._setup_manager_compatibility()

    # Estimate max episode length (MyoSuite environments typically have timeout)
    # Default to 1000 steps if not available
    unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
    max_episode_length = None

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

    # Create mock sim object for viewer compatibility
    # MyoSuite environments have mj_model and mj_data directly on the unwrapped env
    self._mock_sim = self._create_mock_sim()

    # Create mock managers for ManagerBasedRlEnv compatibility (needed for ONNX export)
    self._setup_manager_compatibility()

    # Modify action space if clipping is enabled
    self._modify_action_space()

    # Reset at the start since rsl_rl does not call reset
    obs, _ = self.env.reset()
    self._last_obs_dict = self._convert_obs_to_dict(obs)
    # Verify that observations are on the correct device after initial conversion
    for key, value in self._last_obs_dict.items():
      if isinstance(value, torch.Tensor):
        if value.device != self.device:
          # Force move to correct device immediately
          self._last_obs_dict[key] = value.to(device=self.device)

  def _create_mock_sim(self) -> Any:
    """Create a mock sim object that exposes mj_model, mj_data, and wp_data for viewer compatibility.

    Returns:
      Mock object compatible with both native and Viser viewers
    """
    # Get the underlying MyoSuite environment
    # For SyncVectorEnv, get the first environment
    if isinstance(self.env, vector.VectorEnv):
      if hasattr(self.env, "envs") and len(self.env.envs) > 0:
        myosuite_env = self.env.envs[0]
      elif hasattr(self.env, "single_env"):
        myosuite_env = self.env.single_env
      else:
        # Fallback: try to unwrap
        myosuite_env = getattr(self.env, "unwrapped", None)
    else:
      myosuite_env = self.env

    # Unwrap to get the actual MyoSuite environment
    while (
      myosuite_env is not None
      and hasattr(myosuite_env, "unwrapped")
      and myosuite_env.unwrapped is not myosuite_env
    ):
      myosuite_env = myosuite_env.unwrapped
    if myosuite_env is None:
      raise RuntimeError("Failed to unwrap underlying MyoSuite environment")

    # For mjx/warp versions, check if mj_model and mj_data are accessible
    # They might be accessed differently in mjx/warp versions
    if not hasattr(myosuite_env, "mj_model"):
      # Try alternative access patterns for mjx/warp
      if hasattr(myosuite_env, "model"):
        # Some versions use 'model' instead of 'mj_model'
        myosuite_env.mj_model = myosuite_env.model  # type: ignore[attr-defined]
      elif hasattr(myosuite_env, "_model"):
        myosuite_env.mj_model = myosuite_env._model  # type: ignore[attr-defined]
      else:
        raise RuntimeError(
          "MyoSuite environment does not have mj_model attribute. "
          "This may indicate an incompatible MyoSuite version."
        )

    if not hasattr(myosuite_env, "mj_data"):
      # Try alternative access patterns for mjx/warp
      if hasattr(myosuite_env, "data"):
        myosuite_env.mj_data = myosuite_env.data  # type: ignore[attr-defined]
      elif hasattr(myosuite_env, "_data"):
        myosuite_env.mj_data = myosuite_env._data  # type: ignore[attr-defined]
      else:
        raise RuntimeError(
          "MyoSuite environment does not have mj_data attribute. "
          "This may indicate an incompatible MyoSuite version."
        )

    # Create a mock wp_data object that provides numpy arrays from mj_data
    class MockWpData:
      """Mock wp_data that converts mj_data arrays to the format Viser expects."""

      def __init__(self, env, num_envs: int = 1):
        self._env = env
        self._num_envs = num_envs

      @property
      def _mj_model(self):
        """Access mj_model dynamically from environment, supporting mjx/warp versions."""
        return getattr(self._env, "mj_model", getattr(self._env, "model", None))

      @property
      def _mj_data(self):
        """Access mj_data dynamically from environment, supporting mjx/warp versions."""
        return getattr(self._env, "mj_data", getattr(self._env, "data", None))

      def _to_batched(self, arr: np.ndarray) -> np.ndarray:
        """Convert a 1D or 2D array to batched format (batch_size, ...)"""
        arr = np.asarray(arr)
        if arr.ndim == 0:
          return arr.reshape(1)
        elif arr.ndim == 1:
          return arr[np.newaxis, :]  # (1, n)
        elif arr.ndim == 2:
          return arr[np.newaxis, :, :]  # (1, n, m)
        else:
          return arr[np.newaxis, ...]  # (1, ...)

      def _make_array_proxy(self, data: np.ndarray) -> Any:
        """Create an object with .numpy() method that returns the data."""
        batched = self._to_batched(data)

        class ArrayProxy:
          def __init__(self, arr: np.ndarray):
            self._arr = arr

          def numpy(self) -> np.ndarray:
            return self._arr

        return ArrayProxy(batched)

      @property
      def qpos(self):
        """Joint positions. Shape: (batch_size, nq)"""
        if self._mj_data is None:
          raise RuntimeError("mj_data is not available")
        return self._make_array_proxy(self._mj_data.qpos)

      @property
      def qvel(self):
        """Joint velocities. Shape: (batch_size, nv)"""
        if self._mj_data is None:
          raise RuntimeError("mj_data is not available")
        return self._make_array_proxy(self._mj_data.qvel)

      @property
      def xpos(self):
        """Body positions. Shape: (batch_size, nbody, 3)"""
        # Ensure mj_forward has been called to update xpos
        # MyoSuite environments update mj_data during step, but we need to ensure
        # forward kinematics are computed for visualization
        import mujoco

        if self._mj_model is None or self._mj_data is None:
          raise RuntimeError("mj_model or mj_data is not available")
        mujoco.mj_forward(self._mj_model, self._mj_data)  # type: ignore[attr-defined]
        # xpos is shape (nbody, 3), we need to add batch dimension
        xpos_array = self._mj_data.xpos
        return self._make_array_proxy(xpos_array)

      @property
      def xmat(self):
        """Body orientation matrices. Shape: (batch_size, nbody, 3, 3)"""
        # Ensure mj_forward has been called to update xmat
        import mujoco

        if self._mj_model is None or self._mj_data is None:
          raise RuntimeError("mj_model or mj_data is not available")
        mujoco.mj_forward(self._mj_model, self._mj_data)  # type: ignore[attr-defined]
        # xmat is shape (nbody, 9), reshape to (nbody, 3, 3), then add batch dimension
        xmat_array = self._mj_data.xmat.reshape(-1, 3, 3)
        return self._make_array_proxy(xmat_array)

      @property
      def geom_xpos(self):
        """Geometry positions. Shape: (batch_size, ngeom, 3)"""
        # Ensure mj_forward has been called to update geom_xpos
        import mujoco

        if self._mj_model is None or self._mj_data is None:
          raise RuntimeError("mj_model or mj_data is not available")
        mujoco.mj_forward(self._mj_model, self._mj_data)  # type: ignore[attr-defined]
        # geom_xpos is shape (ngeom, 3), we need to add batch dimension
        return self._make_array_proxy(self._mj_data.geom_xpos)

      @property
      def geom_xmat(self):
        """Geometry orientation matrices. Shape: (batch_size, ngeom, 3, 3)"""
        # Ensure mj_forward has been called to update geom_xmat
        import mujoco

        if self._mj_model is None or self._mj_data is None:
          raise RuntimeError("mj_model or mj_data is not available")
        mujoco.mj_forward(self._mj_model, self._mj_data)  # type: ignore[attr-defined]
        # geom_xmat is shape (ngeom, 9), reshape to (ngeom, 3, 3), then add batch dimension
        return self._make_array_proxy(self._mj_data.geom_xmat.reshape(-1, 3, 3))

      @property
      def mocap_pos(self):
        """Mocap positions. Shape: (batch_size, nmocap, 3)"""
        if self._mj_model is None or self._mj_data is None:
          raise RuntimeError("mj_model or mj_data is not available")
        if self._mj_model.nmocap > 0:
          return self._make_array_proxy(self._mj_data.mocap_pos)
        else:
          # Return empty array with correct shape
          empty = np.zeros((0, 3), dtype=np.float32)
          return self._make_array_proxy(empty)

      @property
      def mocap_quat(self):
        """Mocap quaternions. Shape: (batch_size, nmocap, 4)"""
        if self._mj_model is None or self._mj_data is None:
          raise RuntimeError("mj_model or mj_data is not available")
        if self._mj_model.nmocap > 0:
          return self._make_array_proxy(self._mj_data.mocap_quat)
        else:
          # Return empty array with correct shape
          empty = np.zeros((0, 4), dtype=np.float32)
          return self._make_array_proxy(empty)

    # Create a mock sim object that inherits from Simulation to pass isinstance checks
    # We need to import Simulation here to avoid circular imports
    # Use lazy import to avoid triggering mjlab import chain issues
    try:
      from mjlab.sim.sim import Simulation as _SimulationBase
    except (ImportError, AttributeError):
      # If Simulation import fails, use fallback
      _SimulationBase = object

    class MockSimPrimary(_SimulationBase):  # type: ignore[misc]
      """Mock Simulation that works with MyoSuite environments.

      This class inherits from Simulation to pass isinstance checks,
      but bypasses the normal __init__ to avoid MuJoCo Warp requirements.
      """

      def __init__(self, env):
        # Don't call super().__init__() - we're bypassing MuJoCo Warp setup
        # Instead, set up minimal attributes needed for compatibility
        self._env = (
          env  # This is the wrapped env (SyncVectorEnv), keep for compatibility
        )
        # Get the actual unwrapped MyoSuite environment for model access
        myosuite_env = env
        # Unwrap to get the actual MyoSuite environment
        while (
          myosuite_env is not None
          and hasattr(myosuite_env, "unwrapped")
          and myosuite_env.unwrapped is not myosuite_env
        ):
          myosuite_env = myosuite_env.unwrapped

        # Also check if it's a VectorEnv and get the first env
        if hasattr(myosuite_env, "envs") and len(myosuite_env.envs) > 0:
          myosuite_env = myosuite_env.envs[0]
          while (
            myosuite_env is not None
            and hasattr(myosuite_env, "unwrapped")
            and myosuite_env.unwrapped is not myosuite_env
          ):
            myosuite_env = myosuite_env.unwrapped

        # Store the actual MyoSuite environment for model access
        self._myosuite_env = myosuite_env

        # Support both standard and mjx/warp versions
        # Get model from the actual MyoSuite environment
        self._mj_model = None
        if myosuite_env is not None:
          self._mj_model = getattr(
            myosuite_env, "mj_model", getattr(myosuite_env, "model", None)
          )
          if self._mj_model is None and hasattr(myosuite_env, "sim"):
            self._mj_model = getattr(
              myosuite_env.sim, "mj_model", getattr(myosuite_env.sim, "model", None)
            )

          # NOTE: Textures are defined in the scene XML but MuJoCo doesn't load them
          # into the model (ntext=0) because texture file paths are relative and don't
          # resolve correctly. The snapshot renderer can render textures because it
          # loads them on-demand. For viser to show textures, we need to ensure the
          # model is loaded with textures. Since reloading doesn't work (paths still
          # don't resolve), the issue may be that viser needs the model loaded from
          # XML with the correct working directory, or it needs texture files to be
          # accessible via a different mechanism.

        self._mj_data = None
        if myosuite_env is not None:
          self._mj_data = getattr(
            myosuite_env, "mj_data", getattr(myosuite_env, "data", None)
          )
          if self._mj_data is None and hasattr(myosuite_env, "sim"):
            self._mj_data = getattr(
              myosuite_env.sim, "mj_data", getattr(myosuite_env.sim, "data", None)
            )

        if self._mj_model is None or self._mj_data is None:
          raise RuntimeError(
            "MyoSuite environment must have mj_model/mj_data or model/data attributes. "
            f"Tried to get from: {type(myosuite_env).__name__ if myosuite_env else 'None'}"
          )
        self._wp_data = MockWpData(env, num_envs=1)

        # Set minimal attributes that Simulation expects
        self.num_envs = 1
        self.device = "cpu"  # MyoSuite runs on CPU
        # Set cfg to None or a minimal object if needed
        self.cfg = None

      @property
      def mj_model(self):
        # Return the actual MyoSuite model stored during initialization
        # This is the model from the unwrapped MyoSuite environment, not the wrapper
        return self._mj_model

        @property
        def mj_data(self):
          """Return mj_data, ensuring forward kinematics are up to date."""
          import mujoco

          # Support both standard and mjx/warp versions
          mj_model = getattr(
            self._env,
            "mj_model",
            getattr(self._env, "model", self._mj_model),
          )
          mj_data = getattr(
            self._env, "mj_data", getattr(self._env, "data", self._mj_data)
          )
          mujoco.mj_forward(mj_model, mj_data)  # type: ignore[attr-defined]
          return mj_data

        @property
        def wp_data(self):
          """Return mock wp_data for Viser viewer compatibility."""
          # Ensure forward kinematics are computed before accessing wp_data
          import mujoco

          # Support both standard and mjx/warp versions
          mj_model = getattr(
            self._env,
            "mj_model",
            getattr(self._env, "model", self._mj_model),
          )
          mj_data = getattr(
            self._env, "mj_data", getattr(self._env, "data", self._mj_data)
          )
          mujoco.mj_forward(mj_model, mj_data)  # type: ignore[attr-defined]
          return self._wp_data

        @property
        def data(self):
          """Return data adapter for compatibility with sim.data access.

          OffscreenRenderer expects data.qpos[env_idx].cpu().numpy(), so we need
          to provide a torch tensor interface. This adapter wraps mj_data and
          provides the expected interface.
          """
          import mujoco
          import torch

          # Support both standard and mjx/warp versions
          mj_model = getattr(
            self._env,
            "mj_model",
            getattr(self._env, "model", self._mj_model),
          )
          mj_data = getattr(
            self._env, "mj_data", getattr(self._env, "data", self._mj_data)
          )
          mujoco.mj_forward(mj_model, mj_data)  # type: ignore[attr-defined]

          # Create an adapter that provides torch tensor interface
          class DataAdapter:
            """Adapter to make mj_data look like ManagerBasedRlEnv's sim.data."""

            def __init__(self, mj_data, mj_model):
              self._mj_data = mj_data
              self._mj_model = mj_model
              # nworld is the number of environments (1 for single env)
              self.nworld = 1

            @property
            def qpos(self):
              """Return qpos as torch tensor with batch dimension."""
              # OffscreenRenderer expects data.qpos[env_idx].cpu().numpy()
              # So we need shape (1, nq) for batch_size=1
              qpos_np = self._mj_data.qpos.copy()
              qpos_tensor = torch.from_numpy(qpos_np).unsqueeze(0)  # Add batch dim
              return qpos_tensor

            @property
            def qvel(self):
              """Return qvel as torch tensor with batch dimension."""
              qvel_np = self._mj_data.qvel.copy()
              qvel_tensor = torch.from_numpy(qvel_np).unsqueeze(0)  # Add batch dim
              return qvel_tensor

            @property
            def mocap_pos(self):
              """Return mocap_pos as torch tensor with batch dimension."""
              if self._mj_model.nmocap > 0:
                mocap_pos_np = self._mj_data.mocap_pos.copy()
                return torch.from_numpy(mocap_pos_np).unsqueeze(0)
              else:
                # Return empty tensor with correct shape
                return torch.zeros((1, 0, 3), dtype=torch.float32)

            @property
            def mocap_quat(self):
              """Return mocap_quat as torch tensor with batch dimension."""
              if self._mj_model.nmocap > 0:
                mocap_quat_np = self._mj_data.mocap_quat.copy()
                return torch.from_numpy(mocap_quat_np).unsqueeze(0)
              else:
                # Return empty tensor with correct shape
                return torch.zeros((1, 0, 4), dtype=torch.float32)

          return DataAdapter(mj_data, mj_model)

      # Override methods that might be called but aren't needed
      def create_graph(self) -> None:
        """No-op for MyoSuite (no CUDA graphs needed)."""
        pass

        def forward(self) -> None:
          """Update forward kinematics for visualization."""
          import mujoco

          # Ensure forward kinematics are computed for visualization
          # Support both standard and mjx/warp versions
          mj_model = getattr(
            self._env,
            "mj_model",
            getattr(self._env, "model", self._mj_model),
          )
          mj_data = getattr(
            self._env, "mj_data", getattr(self._env, "data", self._mj_data)
          )
          mujoco.mj_forward(mj_model, mj_data)  # type: ignore[attr-defined]

      def step(self) -> None:
        """No-op for MyoSuite (step handled by MyoSuite)."""
        pass

    # Try to create the mock sim
    try:
      # Create and return the mock sim
      mock_sim = MockSimPrimary(myosuite_env)
    except (TypeError, AttributeError):
      # Fallback: if Simulation import fails or inheritance doesn't work,
      # create a regular class and use __class__ manipulation
      class MockSimFallback:
        def __init__(self, env):
          self._env = env
          # Support both standard and mjx/warp versions
          self._mj_model = getattr(env, "mj_model", getattr(env, "model", None))
          self._mj_data = getattr(env, "mj_data", getattr(env, "data", None))
          if self._mj_model is None or self._mj_data is None:
            raise RuntimeError(
              "MyoSuite environment must have mj_model/mj_data or model/data attributes"
            )
          self._wp_data = MockWpData(env, num_envs=1)
          self.num_envs = 1
          self.device = "cpu"
          self.cfg = None

        @property
        def mj_model(self):
          # Support both standard and mjx/warp versions
          return getattr(
            self._env,
            "mj_model",
            getattr(self._env, "model", self._mj_model),
          )

        @property
        def mj_data(self):
          """Return mj_data, ensuring forward kinematics are up to date."""
          import mujoco

          # Support both standard and mjx/warp versions
          mj_model = getattr(
            self._env,
            "mj_model",
            getattr(self._env, "model", self._mj_model),
          )
          mj_data = getattr(
            self._env, "mj_data", getattr(self._env, "data", self._mj_data)
          )
          mujoco.mj_forward(mj_model, mj_data)
          return mj_data

        @property
        def wp_data(self):
          """Return mock wp_data for Viser viewer compatibility."""
          import mujoco

          # Support both standard and mjx/warp versions
          mj_model = getattr(
            self._env,
            "mj_model",
            getattr(self._env, "model", self._mj_model),
          )
          mj_data = getattr(
            self._env, "mj_data", getattr(self._env, "data", self._mj_data)
          )
          mujoco.mj_forward(mj_model, mj_data)
          return self._wp_data

        @property
        def data(self):
          """Return mj_data for compatibility with sim.data access."""
          import mujoco

          # Support both standard and mjx/warp versions
          mj_model = getattr(
            self._env,
            "mj_model",
            getattr(self._env, "model", self._mj_model),
          )
          mj_data = getattr(
            self._env, "mj_data", getattr(self._env, "data", self._mj_data)
          )
          mujoco.mj_forward(mj_model, mj_data)
          return mj_data

        def create_graph(self) -> None:
          pass

        def forward(self) -> None:
          """Update forward kinematics for visualization."""
          import mujoco

          # Support both standard and mjx/warp versions
          mj_model = getattr(
            self._env,
            "mj_model",
            getattr(self._env, "model", self._mj_model),
          )
          mj_data = getattr(
            self._env, "mj_data", getattr(self._env, "data", self._mj_data)
          )
          if mj_model is not None and mj_data is not None:
            mujoco.mj_forward(mj_model, mj_data)

        def step(self) -> None:
          pass

      mock_sim = MockSimFallback(myosuite_env)

      # Try to make it pass isinstance check by manipulating __class__
      try:
        from mjlab.sim.sim import Simulation

        # Create a dynamic subclass that inherits from Simulation
        class MockSimulationSubclass(Simulation):
          pass

        # Change the instance's class to the subclass
        object.__setattr__(mock_sim, "__class__", MockSimulationSubclass)
      except (ImportError, TypeError, AttributeError):
        # If this fails, the isinstance check in Viser will use the interface check instead
        # This is expected if mjlab is not fully installed or has import issues
        pass

    return mock_sim

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

  def _initialize_offline_renderer(self):
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
      myosuite_env = None
      if isinstance(self.env, vector.VectorEnv):
        if hasattr(self.env, "envs") and len(self.env.envs) > 0:
          myosuite_env = self.env.envs[0]
          while hasattr(myosuite_env, "env") and myosuite_env.env is not myosuite_env:
            myosuite_env = myosuite_env.env
      else:
        myosuite_env = self.env

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
    except Exception:
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
        myosuite_env = None
        if isinstance(self.env, vector.VectorEnv):
          if hasattr(self.env, "envs") and len(self.env.envs) > 0:
            myosuite_env = self.env.envs[0]
            while hasattr(myosuite_env, "env") and myosuite_env.env is not myosuite_env:
              myosuite_env = myosuite_env.env
        else:
          myosuite_env = self.env

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

                  def __init__(self, mj_data, mj_model):
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
                    else:
                      # Return empty tensor with correct shape
                      return torch.zeros((1, 0, 3), dtype=torch.float32)

                  @property
                  def mocap_quat(self):
                    """Return mocap_quat as torch tensor with batch dimension."""
                    if self._mj_model.nmocap > 0:
                      mocap_quat_np = self._mj_data.mocap_quat.copy()
                      return torch.from_numpy(mocap_quat_np).unsqueeze(0)
                    else:
                      # Return empty tensor with correct shape
                      return torch.zeros((1, 0, 4), dtype=torch.float32)

                sim_data = DataAdapter(mj_data, mj_model)

              # Update renderer with sim.data (which should have torch tensor interface)
              self._offline_renderer.update(sim_data)
              frame = self._offline_renderer.render()
              if frame is not None:
                return frame
            except Exception:
              # If update fails, continue to fallback
              pass
      except Exception:
        # If offline renderer fails, try fallback
        pass

    # Fallback: Try underlying environment's render method
    myosuite_env = None
    if isinstance(self.env, vector.VectorEnv):
      if hasattr(self.env, "envs") and len(self.env.envs) > 0:
        myosuite_env = self.env.envs[0]
        while hasattr(myosuite_env, "env") and myosuite_env.env is not myosuite_env:
          myosuite_env = myosuite_env.env
    else:
      myosuite_env = self.env

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
        if isinstance(value, torch.Tensor):
          if value.device != self.device:
            # Force move to correct device immediately
            self._last_obs_dict[key] = value.to(device=self.device)

    # Rebuild observation dict, ensuring all tensors are on the correct device
    # This is necessary because tensors might have been created on CPU initially
    obs_dict_on_device = {}
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

    # CRITICAL: RSL-RL accesses obs['policy'] directly, so we MUST ensure it's on correct device
    # Force move 'policy' observation to correct device with explicit verification
    if "policy" in td:
      policy_val = td["policy"]
      if isinstance(policy_val, torch.Tensor):
        # Create a completely new tensor on the target device
        # Use .clone() to ensure it's a new tensor, then .to() to move it
        policy_on_device = policy_val.clone().to(device=self.device, non_blocking=False)
        # Verify it's actually on the device
        if policy_on_device.device != self.device:
          # If still not on device, force it using CUDA or CPU explicit placement
          if str(self.device).startswith("cuda"):
            device_idx = self.device.index if hasattr(self.device, "index") else 0
            policy_on_device = policy_on_device.cuda(device=device_idx)  # type: ignore[arg-type]
          else:
            policy_on_device = policy_on_device.cpu()
        td["policy"] = policy_on_device.contiguous()

      # Also ensure 'critic' is on correct device (same as policy for MyoSuite)
      if "critic" in td:
        critic_val = td["critic"]
        if isinstance(critic_val, torch.Tensor):
          critic_on_device = critic_val.clone().to(
            device=self.device, non_blocking=False
          )
          if critic_on_device.device != self.device:
            if str(self.device).startswith("cuda"):
              device_idx = self.device.index if hasattr(self.device, "index") else 0
              critic_on_device = critic_on_device.cuda(device=device_idx)  # type: ignore[arg-type]
            else:
              critic_on_device = critic_on_device.cpu()
          td["critic"] = critic_on_device.contiguous()

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
      if isinstance(value, torch.Tensor):
        if value.device != self.device:
          obs_dict[key] = value.to(device=self.device)
    self._last_obs_dict = obs_dict
    self.episode_length_buf.zero_()

    return TensorDict(obs_dict, batch_size=[self.num_envs]), info

  def step(  # type: ignore[override]
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    """Step the environment.

    Returns 4 values for rsl_rl compatibility: (obs, rewards, dones, extras)
    where dones = terminated | truncated.
    """
    # Convert actions to numpy
    if isinstance(actions, torch.Tensor):
      actions_np = actions.cpu().numpy()
    else:
      actions_np = actions

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

    # Update forward kinematics for visualization
    # This ensures the viewer has current position data
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
      if isinstance(value, torch.Tensor):
        if value.device != self.device:
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

  def _convert_obs_to_dict(self, obs: Any) -> dict[str, Any]:
    """Convert observations to dictionary of torch tensors.

    RSL-RL expects observation groups like 'policy' and 'critic'.
    For MyoSuite, we map all observations to both 'policy' and 'critic' groups
    (since we don't have privileged information for critic).
    """
    if isinstance(obs, dict):
      obs_dict = {}
      policy_tensors = []
      for key, value in obs.items():
        # Create tensor and ensure it's on the correct device
        # Always create on target device explicitly to avoid any device issues
        if isinstance(value, torch.Tensor):
          # If already a tensor, ensure it's on the correct device
          if value.device != self.device:
            tensor = value.to(device=self.device).to(dtype=torch.float32)
          else:
            tensor = value.to(dtype=torch.float32)
        elif isinstance(value, np.ndarray):
          # For numpy arrays, create directly on target device
          # Use from_numpy + to() to ensure proper device placement
          tensor = torch.from_numpy(value).to(device=self.device, dtype=torch.float32)
        else:
          # For other types, convert to tensor on target device
          tensor = torch.tensor(value, device=self.device, dtype=torch.float32)
        # Map to 'policy' group for RSL-RL compatibility
        # If key is already 'policy' or 'critic', keep it; otherwise collect for policy
        if key in ["policy", "critic"]:
          obs_dict[key] = tensor
        else:
          # For MyoSuite observations, collect them to concatenate into 'policy' group
          policy_tensors.append(tensor)
      # Concatenate all non-policy/critic observations into a single 'policy' tensor
      if policy_tensors:
        policy_obs = (
          torch.cat(policy_tensors, dim=-1)
          if len(policy_tensors) > 1
          else policy_tensors[0]
        )
        obs_dict["policy"] = policy_obs
        # Also provide 'critic' observation (same as policy for MyoSuite)
        if "critic" not in obs_dict:
          obs_dict["critic"] = policy_obs
      return obs_dict
    elif isinstance(obs, np.ndarray):
      # Single observation array - map to both 'policy' and 'critic' groups for RSL-RL
      # Create tensor directly on the correct device using from_numpy + to()
      policy_obs = torch.from_numpy(obs).to(device=self.device, dtype=torch.float32)
      return {"policy": policy_obs, "critic": policy_obs}
    elif isinstance(obs, (list, tuple)):
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
      else:
        # List of arrays - map to both 'policy' and 'critic' groups
        policy_obs = torch.from_numpy(np.array(obs)).to(
          device=self.device, dtype=torch.float32
        )
        return {"policy": policy_obs, "critic": policy_obs}
    else:
      # Fallback - try to convert directly, map to both 'policy' and 'critic' groups
      # Convert to numpy first if not already, then use from_numpy for proper device placement
      if isinstance(obs, np.ndarray):
        policy_obs = torch.from_numpy(obs).to(device=self.device, dtype=torch.float32)
      else:
        policy_obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
      return {"policy": policy_obs, "critic": policy_obs}

  def _setup_manager_compatibility(self):
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
