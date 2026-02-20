"""Simulation compatibility helpers for MyoSuite-based environments.

This module provides utilities to create a mock `Simulation`-like object that
exposes `mj_model`, `mj_data`, `wp_data`, and `data` attributes in a form
compatible with mjlab's viewers (including Viser and offscreen rendering).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import vector


def create_mock_sim(env: Any) -> Any:
  """Create a mock sim object that exposes mj_model, mj_data, and wp_data.

  The returned object is compatible with both native and Viser viewers and,
  when mjlab is available, attempts to inherit from `mjlab.sim.sim.Simulation`
  so that `isinstance(sim, Simulation)` checks pass.
  """
  # Get the underlying MyoSuite environment
  # For SyncVectorEnv or other VectorEnv types, get the first environment
  if isinstance(env, vector.VectorEnv):
    if hasattr(env, "envs") and len(env.envs) > 0:
      myosuite_env = env.envs[0]
    elif hasattr(env, "single_env"):
      myosuite_env = env.single_env
    else:
      # Fallback: try to unwrap
      myosuite_env = getattr(env, "unwrapped", None)
  else:
    myosuite_env = env

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

    def __init__(self, env: Any, num_envs: int = 1):
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
      """Convert a 1D or 2D array to batched format (batch_size, ...)."""
      arr = np.asarray(arr)
      if arr.ndim == 0:
        return arr.reshape(1)
      if arr.ndim == 1:
        return arr[np.newaxis, :]  # (1, n)
      if arr.ndim == 2:
        return arr[np.newaxis, :, :]  # (1, n, m)
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
      """Joint positions. Shape: (batch_size, nq)."""
      if self._mj_data is None:
        raise RuntimeError("mj_data is not available")
      return self._make_array_proxy(self._mj_data.qpos)

    @property
    def qvel(self):
      """Joint velocities. Shape: (batch_size, nv)."""
      if self._mj_data is None:
        raise RuntimeError("mj_data is not available")
      return self._make_array_proxy(self._mj_data.qvel)

    @property
    def xpos(self):
      """Body positions. Shape: (batch_size, nbody, 3)."""
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
      """Body orientation matrices. Shape: (batch_size, nbody, 3, 3)."""
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
      """Geometry positions. Shape: (batch_size, ngeom, 3)."""
      # Ensure mj_forward has been called to update geom_xpos
      import mujoco

      if self._mj_model is None or self._mj_data is None:
        raise RuntimeError("mj_model or mj_data is not available")
      mujoco.mj_forward(self._mj_model, self._mj_data)  # type: ignore[attr-defined]
      # geom_xpos is shape (ngeom, 3), we need to add batch dimension
      return self._make_array_proxy(self._mj_data.geom_xpos)

    @property
    def geom_xmat(self):
      """Geometry orientation matrices. Shape: (batch_size, ngeom, 3, 3)."""
      # Ensure mj_forward has been called to update geom_xmat
      import mujoco

      if self._mj_model is None or self._mj_data is None:
        raise RuntimeError("mj_model or mj_data is not available")
      mujoco.mj_forward(self._mj_model, self._mj_data)  # type: ignore[attr-defined]
      # geom_xmat is shape (ngeom, 9), reshape to (ngeom, 3, 3), then add batch dimension
      return self._make_array_proxy(self._mj_data.geom_xmat.reshape(-1, 3, 3))

    @property
    def mocap_pos(self):
      """Mocap positions. Shape: (batch_size, nmocap, 3)."""
      if self._mj_model is None or self._mj_data is None:
        raise RuntimeError("mj_model or mj_data is not available")
      if self._mj_model.nmocap > 0:
        return self._make_array_proxy(self._mj_data.mocap_pos)
      # Return empty array with correct shape
      empty = np.zeros((0, 3), dtype=np.float32)
      return self._make_array_proxy(empty)

    @property
    def mocap_quat(self):
      """Mocap quaternions. Shape: (batch_size, nmocap, 4)."""
      if self._mj_model is None or self._mj_data is None:
        raise RuntimeError("mj_model or mj_data is not available")
      if self._mj_model.nmocap > 0:
        return self._make_array_proxy(self._mj_data.mocap_quat)
      # Return empty array with correct shape
      empty = np.zeros((0, 4), dtype=np.float32)
      return self._make_array_proxy(empty)

  # Create a mock sim object that inherits from Simulation to pass isinstance checks
  # We need to import Simulation here to avoid circular imports
  # Use lazy import to avoid triggering mjlab import chain issues
  try:
    from mjlab.sim.sim import Simulation as _SimulationBase
  except (ImportError, AttributeError):  # pragma: no cover - defensive fallback
    # If Simulation import fails, use fallback
    _SimulationBase = object

  class MockSimPrimary(_SimulationBase):  # type: ignore[misc]
    """Mock Simulation that works with MyoSuite environments.

    This class inherits from Simulation to pass isinstance checks,
    but bypasses the normal __init__ to avoid MuJoCo Warp requirements.
    """

    def __init__(self, env: Any):
      # Don't call super().__init__() - we're bypassing MuJoCo Warp setup
      # Instead, set up minimal attributes needed for compatibility
      self._env = env  # This is the wrapped env (or underlying MyoSuite env)
      # Get the actual unwrapped MyoSuite environment for model access
      inner_env = env
      # Unwrap to get the actual MyoSuite environment
      while (
        inner_env is not None
        and hasattr(inner_env, "unwrapped")
        and inner_env.unwrapped is not inner_env
      ):
        inner_env = inner_env.unwrapped

      # Also check if it's a VectorEnv and get the first env
      if hasattr(inner_env, "envs") and len(inner_env.envs) > 0:
        inner_env = inner_env.envs[0]
        while (
          inner_env is not None
          and hasattr(inner_env, "unwrapped")
          and inner_env.unwrapped is not inner_env
        ):
          inner_env = inner_env.unwrapped

      # Store the actual MyoSuite environment for model access
      self._myosuite_env = inner_env

      # Support both standard and mjx/warp versions
      # Get model from the actual MyoSuite environment
      self._mj_model = None
      if inner_env is not None:
        self._mj_model = getattr(
          inner_env, "mj_model", getattr(inner_env, "model", None)
        )
        if self._mj_model is None and hasattr(inner_env, "sim"):
          self._mj_model = getattr(
            inner_env.sim, "mj_model", getattr(inner_env.sim, "model", None)
          )

      self._mj_data = None
      if inner_env is not None:
        self._mj_data = getattr(inner_env, "mj_data", getattr(inner_env, "data", None))
        if self._mj_data is None and hasattr(inner_env, "sim"):
          self._mj_data = getattr(
            inner_env.sim, "mj_data", getattr(inner_env.sim, "data", None)
          )

      if self._mj_model is None or self._mj_data is None:
        raise RuntimeError(
          "MyoSuite environment must have mj_model/mj_data or model/data attributes. "
          f"Tried to get from: {type(inner_env).__name__ if inner_env else 'None'}"
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
      mj_data = getattr(self._env, "mj_data", getattr(self._env, "data", self._mj_data))
      mujoco.mj_forward(mj_model, mj_data)  # type: ignore[attr-defined]
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
      mj_data = getattr(self._env, "mj_data", getattr(self._env, "data", self._mj_data))
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
      mj_data = getattr(self._env, "mj_data", getattr(self._env, "data", self._mj_data))
      mujoco.mj_forward(mj_model, mj_data)  # type: ignore[attr-defined]

      # Create an adapter that provides torch tensor interface
      class DataAdapter:
        """Adapter to make mj_data look like ManagerBasedRlEnv's sim.data."""

        def __init__(self, mj_data: Any, mj_model: Any):
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

      return DataAdapter(mj_data, mj_model)

    # Override methods that might be called but aren't needed
    def create_graph(self) -> None:
      """No-op for MyoSuite (no CUDA graphs needed)."""

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
      mj_data = getattr(self._env, "mj_data", getattr(self._env, "data", self._mj_data))
      mujoco.mj_forward(mj_model, mj_data)  # type: ignore[attr-defined]

    def step(self) -> None:
      """No-op for MyoSuite (step handled by MyoSuite)."""

  # Try to create the mock sim
  try:
    # Create and return the mock sim
    mock_sim = MockSimPrimary(myosuite_env)
  except (TypeError, AttributeError):
    # Fallback: if Simulation import fails or inheritance doesn't work,
    # create a regular class and use __class__ manipulation
    class MockSimFallback:
      def __init__(self, env: Any):
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
        """No-op for MyoSuite."""

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
        """No-op for MyoSuite."""

    mock_sim = MockSimFallback(myosuite_env)

    # Try to make it pass isinstance check by manipulating __class__
    try:  # pragma: no cover - best-effort enhancement
      from mjlab.sim.sim import Simulation

      # Create a dynamic subclass that inherits from Simulation
      class MockSimulationSubclass(Simulation):  # type: ignore[misc]
        pass

      # Change the instance's class to the subclass
      object.__setattr__(mock_sim, "__class__", MockSimulationSubclass)
    except (ImportError, TypeError, AttributeError):
      # If this fails, the isinstance check in Viser will use the interface check instead
      # This is expected if mjlab is not fully installed or has import issues
      pass

  return mock_sim
