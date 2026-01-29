"""Wrapper script for mjlab native play script with MyoSuite environment registration.

This script ensures MyoSuite environments are registered before mjlab's native
play script runs, allowing MyoSuite tasks to be used with mjlab's native CLI.

For MyoSuite tasks, this script patches mjlab's run_play to use gym.make()
instead of ManagerBasedRlEnv, inheriting all other logic from mjlab.
"""

# ruff: noqa: E402, I001

import time

# Import mjlab_myosuite FIRST to trigger auto-registration of MyoSuite environments
# This MUST happen before any mjlab imports to ensure registration completes
# before tyro evaluates choices
try:
  # Force registration to complete by accessing the registry
  import gymnasium as gym

  import mjlab_myosuite  # noqa: F401

  # Trigger registration multiple times to ensure it completes
  for _ in range(3):
    _ = list(gym.registry.keys())  # Trigger any lazy registration
    time.sleep(0.1)

  # Verify MyoSuite environments are registered
  myosuite_tasks = [k for k in gym.registry.keys() if "Mjlab-MyoSuite" in k]
  if myosuite_tasks:
    print(f"[INFO] Registered {len(myosuite_tasks)} MyoSuite environments")
except ImportError:
  pass  # MyoSuite not available, skip registration
except Exception as e:
  # Log but don't fail - registration might have partially completed
  import warnings

  warnings.warn(f"MyoSuite registration warning: {e}", UserWarning, stacklevel=2)

# Now import mjlab's play module and patch run_play for MyoSuite tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg as mjlab_load_env_cfg

from mjlab.scripts import play as mjlab_play_module

# Patch mjlab primitive creation so dynamic BOX geoms (dice) get textures.
# This allows ViserPlayViewer to render the dice the same way as the minimal test.
try:
  import mujoco as _mujoco  # noqa: F401

  from mjlab.viewer.viser import conversions as _viser_conversions
  from mjlab_myosuite.viewer_helpers import create_textured_dice_box_mesh

  if not getattr(_viser_conversions, "_myosuite_box_patch", False):
    _orig_create_primitive_mesh = _viser_conversions.create_primitive_mesh

    def _patched_create_primitive_mesh(mj_model, geom_id):
      # If this geom is a textured box, return a textured dice mesh (local coords).
      try:
        if mj_model.geom_type[geom_id] == _mujoco.mjtGeom.mjGEOM_BOX:
          # Only patch when a texture exists.
          matid = int(mj_model.geom_matid[geom_id])
          texid = -1
          if 0 <= matid < mj_model.nmat:
            rgb = int(mj_model.mat_texid[matid, _mujoco.mjtTextureRole.mjTEXROLE_RGB])
            rgba_tex = int(
              mj_model.mat_texid[matid, _mujoco.mjtTextureRole.mjTEXROLE_RGBA]
            )
            texid = rgb if rgb >= 0 else rgba_tex
          if texid >= 0:
            mesh = create_textured_dice_box_mesh(
              mj_model, geom_id, bake_transform=False
            )
            if mesh is not None:
              return mesh
      except Exception:
        pass

      return _orig_create_primitive_mesh(mj_model, geom_id)

    _viser_conversions.create_primitive_mesh = _patched_create_primitive_mesh
    _viser_conversions._myosuite_box_patch = True
except Exception:
  # If patching fails, fall back to mjlab default behavior.
  pass

# Store the original functions
_original_run_play = mjlab_play_module.run_play
_original_load_env_cfg = mjlab_load_env_cfg
_original_manager_init = ManagerBasedRlEnv.__init__


def _patched_load_env_cfg(task_name: str, play: bool = False):
  """Patched load_env_cfg that adds mock scene for MyoSuite configs."""
  cfg = _original_load_env_cfg(task_name, play)

  # Check if this is a MyoSuite config without a scene
  if task_name.startswith("Mjlab-MyoSuite"):
    try:
      from mjlab_myosuite.config import MyoSuiteEnvCfg
      from mjlab_myosuite.tasks.tracking.tracking_env_cfg import (
        MyoSuiteTrackingEnvCfg,
      )

      if isinstance(cfg, (MyoSuiteEnvCfg, MyoSuiteTrackingEnvCfg)):
        # Ensure scene is set
        if cfg.scene is None:
          try:
            from mjlab.scene.scene import SceneCfg

            scene_cfg = SceneCfg(
              num_envs=getattr(cfg, "num_envs", 1),
              entities={},
              sensors=(),  # tuple, not dict
              extent=None,
              terrain=None,
              env_spacing=2.0,
              spec_fn=None,
            )
            # Use object.__setattr__ for frozen dataclass
            object.__setattr__(cfg, "scene", scene_cfg)

          except (ImportError, Exception):
            print("[WARNING] Failed to create scene cfg and falling back to mock")

            # Fallback: create a simple mock that matches SceneCfg interface
            class _MockSceneCfg:
              def __init__(self, num_envs=1):
                self.num_envs = num_envs
                self.extent = None
                self.entities = {}
                self.sensors = ()  # tuple
                self.terrain = None
                self.env_spacing = 2.0
                self.spec_fn = None

            num_envs = getattr(cfg, "num_envs", 1)
            object.__setattr__(cfg, "scene", _MockSceneCfg(num_envs))

        # Ensure sim is set (needed for ManagerBasedRlEnv)
        if cfg.sim is None:
          try:
            from mjlab.sim.sim import SimulationCfg

            # Use object.__setattr__ for frozen dataclass
            object.__setattr__(cfg, "sim", SimulationCfg())
          except (ImportError, Exception):
            # Fallback: create a minimal mock
            class _MockSimCfg:
              def __init__(self):
                self.mujoco = None

            object.__setattr__(cfg, "sim", _MockSimCfg())

        # Ensure viewer is set (needed for viewer initialization)
        if cfg.viewer is None:
          try:
            from mjlab.viewer import ViewerConfig

            object.__setattr__(cfg, "viewer", ViewerConfig())
          except (ImportError, Exception):
            # Fallback: create a minimal mock
            class _MockViewerCfg:
              def __init__(self):
                self.env_idx = 0
                self.height = 480
                self.width = 640
                self.distance = 5.0
                self.azimuth = 45.0
                self.elevation = -30.0

            object.__setattr__(cfg, "viewer", _MockViewerCfg())
    except ImportError:
      pass  # MyoSuite not available

  return cfg


# Patch ManagerBasedRlEnv to handle MyoSuite configs
# For MyoSuite tasks, ManagerBasedRlEnv IS used, but it's patched to wrap the
# underlying MyoSuite Gymnasium environment. This allows MyoSuite tasks to work
# with mjlab's native training pipeline while preserving MyoSuite functionality.
def _patched_manager_init(self, cfg, device, render_mode=None):
  """Patched ManagerBasedRlEnv.__init__ that wraps MyoSuite environments.

  For MyoSuite configs, this creates a MyoSuite environment via gym.make() and
  configures ManagerBasedRlEnv to wrap it. The MyoSuite environment is stored in
  self.myosuite_env for use by observation/action terms.
  """
  # Check if this is a MyoSuite config
  try:
    from mjlab_myosuite.config import MyoSuiteEnvCfg
    from mjlab_myosuite.tasks.tracking.tracking_env_cfg import (
      MyoSuiteTrackingEnvCfg,
    )

    if isinstance(cfg, (MyoSuiteEnvCfg, MyoSuiteTrackingEnvCfg)):
      # Get task_id from config (set by _patched_run_train)
      task_id = getattr(cfg, "task_id", None)
      if task_id is None:
        # Try to infer from config attributes or raise error
        raise RuntimeError(
          "task_id not set in MyoSuiteEnvCfg. "
          "This should be set automatically by _patched_run_train."
        )

      # Ensure scene is configured BEFORE creating MyoSuite environment
      # Scene needs to exist before ManagerBasedRlEnv.__init__ is called
      # Always set scene (even if it's already set, ensure it's valid)
      # Check if scene is None or invalid
      if cfg.scene is None or not hasattr(cfg.scene, "extent"):
        try:
          from mjlab.scene.scene import SceneCfg

          # Create a scene config that will use MyoSuite's model
          scene_cfg = SceneCfg(
            num_envs=getattr(cfg, "num_envs", 1),
            entities={},  # Empty - we'll use MyoSuite's model
            sensors=(),  # tuple, not dict
            extent=None,
            terrain=None,
            env_spacing=2.0,
            spec_fn=None,  # Will be set to use MyoSuite's model after env creation
          )
          object.__setattr__(cfg, "scene", scene_cfg)
        except (ImportError, Exception):
          # Fallback: create a simple mock that matches SceneCfg interface
          class _MockSceneCfg:
            def __init__(self, num_envs=1):
              self.num_envs = num_envs
              self.extent = None
              self.entities = {}
              self.sensors = ()  # tuple
              self.terrain = None
              self.env_spacing = 2.0
              self.spec_fn = None

          num_envs = getattr(cfg, "num_envs", 1)
          object.__setattr__(cfg, "scene", _MockSceneCfg(num_envs))

      # Create MyoSuite environment first (needed for observation/action terms)
      # We need to create it before setting up observations/actions so we can
      # reference it in the observation/action term configs
      import gymnasium as gym

      myosuite_env = gym.make(
        task_id,
        cfg=cfg,
        device=device,
        render_mode=render_mode,
      )

      # Store MyoSuite environment for use by observation/action terms
      # We'll set this after super().__init__ completes
      self._myosuite_env = myosuite_env

      # Configure observations to extract from MyoSuite
      # This MUST be done BEFORE _original_manager_init is called
      # because the observation manager is created during initialization
      if not cfg.observations:
        try:
          from mjlab.managers.observation_group import ObservationGroupCfg

          from mjlab_myosuite.managers import MyoSuiteObservationTermCfg

          if MyoSuiteObservationTermCfg is not None:
            cfg.observations = {
              "policy": ObservationGroupCfg(
                terms={"myosuite_obs": MyoSuiteObservationTermCfg()}
              )
            }
          else:
            # Fallback: create empty observations (will fail later but at least won't crash here)
            cfg.observations = {}
        except ImportError:
          # mjlab managers not available
          cfg.observations = {}

      # Configure actions to pass through to MyoSuite
      # This MUST be done BEFORE _original_manager_init is called
      if not cfg.actions:
        try:
          from mjlab.managers.action_term import ActionTermCfg  # noqa: F401

          from mjlab_myosuite.managers import MyoSuiteActionTermCfg

          if MyoSuiteActionTermCfg is not None:
            cfg.actions = {"joint_pos": MyoSuiteActionTermCfg()}
          else:
            # Fallback: create empty actions
            cfg.actions = {}
        except ImportError:
          # mjlab managers not available
          cfg.actions = {}

      # Ensure sim and viewer are set
      if cfg.sim is None:
        from mjlab.sim.sim import SimulationCfg

        object.__setattr__(cfg, "sim", SimulationCfg())
      if cfg.viewer is None:
        from mjlab.viewer import ViewerConfig

        object.__setattr__(cfg, "viewer", ViewerConfig())

  except (ImportError, AttributeError) as e:
    # Not a MyoSuite config or MyoSuite not available
    # If it's a MyoSuite config but setup failed, we need to handle it
    if isinstance(cfg, (MyoSuiteEnvCfg, MyoSuiteTrackingEnvCfg)):
      raise RuntimeError(
        f"Failed to set up ManagerBasedRlEnv for MyoSuite config: {e}"
      ) from e

  # Call original initialization
  # Type ignore: MyoSuite configs are patched to work with ManagerBasedRlEnv
  result = _original_manager_init(self, cfg, device, render_mode)  # type: ignore[arg-type]

  # After initialization, store MyoSuite environment if we created one
  if hasattr(self, "_myosuite_env"):
    self.myosuite_env = self._myosuite_env
    del self._myosuite_env

    # Override Scene to use MyoSuite's model
    # The Scene was already created, but we can patch it to use MyoSuite's model
    if hasattr(self, "scene") and hasattr(self, "sim"):
      # Get the MyoSuite environment's model
      myosuite_model = None
      if hasattr(self.myosuite_env, "mj_model"):
        myosuite_model = self.myosuite_env.mj_model
      elif hasattr(self.myosuite_env, "unwrapped"):
        unwrapped = self.myosuite_env.unwrapped
        if hasattr(unwrapped, "mj_model"):
          myosuite_model = unwrapped.mj_model
        elif hasattr(unwrapped, "env"):
          # For MyoSuiteVecEnvWrapper, get from underlying env
          if hasattr(unwrapped.env, "mj_model"):
            myosuite_model = unwrapped.env.mj_model

      if myosuite_model is not None:
        # Replace sim's model with MyoSuite's model
        self.sim.mj_model = myosuite_model
        # Also update the scene's compiled model if it exists
        if hasattr(self.scene, "_compiled_model"):
          self.scene._compiled_model = myosuite_model

    # Add get_observations() method that delegates to MyoSuite environment
    # This is needed for rsl_rl runners that expect get_observations()
    # ManagerBasedRlEnv might already have get_observations from VecEnv, but we override it
    # to ensure it uses MyoSuite environment's observations
    def get_observations(self):
      """Get observations from MyoSuite environment.

      This method delegates to the wrapped MyoSuite environment to get observations.
      We need to find the MyoSuiteVecEnvWrapper in the wrapper chain.
      """
      if hasattr(self, "myosuite_env"):
        # Find MyoSuiteVecEnvWrapper in the wrapper chain
        from mjlab_myosuite.wrapper import MyoSuiteVecEnvWrapper

        current = self.myosuite_env
        myosuite_wrapper = None
        max_depth = 10
        depth = 0

        while depth < max_depth and current is not None:
          if isinstance(current, MyoSuiteVecEnvWrapper):
            myosuite_wrapper = current
            break

          # Try to unwrap
          if hasattr(current, "unwrapped"):
            current = current.unwrapped
          elif hasattr(current, "env"):
            current = current.env
          else:
            break
          depth += 1

        if myosuite_wrapper is not None and hasattr(
          myosuite_wrapper, "get_observations"
        ):
          # Get observations from MyoSuiteVecEnvWrapper
          # This handles reset if needed
          obs = myosuite_wrapper.get_observations()
          # Ensure observations are in the correct format (TensorDict with 'policy' key)
          from tensordict import TensorDict

          if isinstance(obs, TensorDict):
            # If it's already a TensorDict, ensure it has 'policy' key
            if "policy" not in obs and len(obs) > 0:
              # Use the first key as 'policy'
              first_key = list(obs.keys())[0]
              obs = TensorDict({"policy": obs[first_key]}, batch_size=obs.batch_size)
            elif "policy" not in obs:
              # Empty observations - this shouldn't happen but handle it
              # Reset the environment to get initial observations
              reset_obs, _ = myosuite_wrapper.env.reset()
              obs_dict = myosuite_wrapper._convert_obs_to_dict(reset_obs)
              obs = TensorDict(obs_dict, batch_size=[myosuite_wrapper.num_envs])
          return obs
        else:
          # Fallback: use observation manager if available
          if hasattr(self, "observation_manager") and hasattr(self, "obs_buf"):
            from tensordict import TensorDict

            # Convert obs_buf to TensorDict format
            obs_dict = {}
            for key, value in self.obs_buf.items():
              obs_dict[key] = value
            return TensorDict(obs_dict, batch_size=[self.num_envs])
          else:
            raise RuntimeError(
              "Cannot get observations: MyoSuiteVecEnvWrapper not found in wrapper chain "
              "and observation_manager is not available"
            )
      else:
        raise RuntimeError(
          "Cannot get observations: ManagerBasedRlEnv has no myosuite_env attribute"
        )

    # Bind the method to the instance (always override to use MyoSuite env)
    import types

    self.get_observations = types.MethodType(get_observations, self)

    # Add num_actions attribute that delegates to MyoSuite environment
    # This is needed for rsl_rl runners
    if not hasattr(self, "num_actions"):
      # Find MyoSuiteVecEnvWrapper to get num_actions
      from mjlab_myosuite.wrapper import MyoSuiteVecEnvWrapper

      current = self.myosuite_env
      myosuite_wrapper = None
      max_depth = 10
      depth = 0

      while depth < max_depth and current is not None:
        if isinstance(current, MyoSuiteVecEnvWrapper):
          myosuite_wrapper = current
          break

        # Try to unwrap
        if hasattr(current, "unwrapped"):
          current = current.unwrapped
        elif hasattr(current, "env"):
          current = current.env
        else:
          break
        depth += 1

      if myosuite_wrapper is not None and hasattr(myosuite_wrapper, "num_actions"):
        self.num_actions = myosuite_wrapper.num_actions
      elif hasattr(self, "action_space"):
        # Fallback: compute from action space
        import numpy as np

        if hasattr(self.action_space, "shape"):
          self.num_actions = int(np.prod(self.action_space.shape))
        else:
          self.num_actions = 1

      # Set max_episode_length via cfg.episode_length_s if available
      # ManagerBasedRlEnv computes max_episode_length from cfg.episode_length_s
      if myosuite_wrapper is not None and hasattr(
        myosuite_wrapper, "max_episode_length"
      ):
        # Set episode_length_s in config so ManagerBasedRlEnv computes max_episode_length correctly
        max_ep_len = myosuite_wrapper.max_episode_length
        if hasattr(cfg, "episode_length_s") and cfg.episode_length_s == 0.0:
          # Compute episode_length_s from max_episode_length and step_dt
          # ManagerBasedRlEnv uses: max_episode_length = int(cfg.episode_length_s / self.step_dt)
          # So: episode_length_s = max_episode_length * step_dt
          # Default step_dt is usually 0.02 (50 Hz)
          step_dt = getattr(cfg, "decimation", 1) * 0.002  # Default MuJoCo timestep
          if hasattr(self, "step_dt"):
            step_dt = self.step_dt
          elif hasattr(cfg, "sim") and hasattr(cfg.sim, "mujoco"):
            step_dt = cfg.sim.mujoco.timestep * getattr(cfg, "decimation", 1)
          episode_length_s = max_ep_len * step_dt
          object.__setattr__(cfg, "episode_length_s", episode_length_s)

    # Override step() method to bypass action manager and delegate to MyoSuite environment
    # This is needed because the action manager has 0 active terms (ActionTermCfg not available)
    import torch

    original_step = self.step

    def step(self, actions: torch.Tensor):
      """Step the environment, bypassing action manager for MyoSuite."""
      # Find MyoSuiteVecEnvWrapper in the wrapper chain
      from mjlab_myosuite.wrapper import MyoSuiteVecEnvWrapper

      current = self.myosuite_env
      myosuite_wrapper = None
      max_depth = 10
      depth = 0

      while depth < max_depth and current is not None:
        if isinstance(current, MyoSuiteVecEnvWrapper):
          myosuite_wrapper = current
          break

        # Try to unwrap
        if hasattr(current, "unwrapped"):
          current = current.unwrapped
        elif hasattr(current, "env"):
          current = current.env
        else:
          break
        depth += 1

      if myosuite_wrapper is not None:
        # Step the MyoSuite environment directly
        obs, rewards, dones, infos = myosuite_wrapper.step(actions)

        # Convert to ManagerBasedRlEnv format
        from tensordict import TensorDict

        if isinstance(obs, TensorDict):
          obs_dict = obs
        elif isinstance(obs, dict):
          obs_dict = TensorDict(obs, batch_size=[self.num_envs])
        else:
          obs_dict = TensorDict({"policy": obs}, batch_size=[self.num_envs])

        # Convert rewards and dones to tensors on correct device
        if isinstance(rewards, torch.Tensor):
          rewards_t = rewards.to(self.device)
        else:
          rewards_t = torch.tensor(rewards, device=self.device, dtype=torch.float32)

        if isinstance(dones, torch.Tensor):
          dones_t = dones.to(self.device)
        else:
          dones_t = torch.tensor(dones, device=self.device, dtype=torch.bool)

        # Update episode_length_buf
        self.episode_length_buf += 1
        reset_mask = dones_t
        if reset_mask.any():
          reset_ids = reset_mask.nonzero(as_tuple=False).flatten()
          self.episode_length_buf[reset_ids] = 0

        # Update observation buffer for observation manager compatibility
        if hasattr(self, "obs_buf"):
          # Store observations in obs_buf format expected by observation manager
          for key, value in obs_dict.items():
            self.obs_buf[key] = value

        return obs_dict, rewards_t, dones_t, infos
      else:
        # Fallback to original step (shouldn't happen)
        return original_step(actions)

    # Bind the step method
    import types

    self.step = types.MethodType(step, self)

  return result


# Patch ManagerBasedRlEnv
ManagerBasedRlEnv.__init__ = _patched_manager_init

# Patch load_env_cfg in both the registry module and play module
# This ensures it's patched regardless of where it's imported from
try:
  from mjlab.tasks import registry as mjlab_registry

  mjlab_registry.load_env_cfg = _patched_load_env_cfg
except ImportError:
  pass

# Also patch it in the play module if it has its own reference
if hasattr(mjlab_play_module, "load_env_cfg"):
  mjlab_play_module.load_env_cfg = _patched_load_env_cfg


def _patched_run_play(task_id: str, cfg) -> None:
  """Patched run_play that uses gym.make() for MyoSuite tasks.

  This function inherits all logic from mjlab's original run_play, but
  uses gym.make() directly for MyoSuite tasks instead of ManagerBasedRlEnv.
  """
  # Check if this is a MyoSuite task
  if task_id.startswith("Mjlab-MyoSuite"):
    # Import MyoSuite-specific components and mjlab utilities
    try:
      from pathlib import Path

      import gymnasium as gym
      import torch
      from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
      from mjlab.utils.torch import configure_torch_backends

      from mjlab_myosuite.config import MyoSuiteEnvCfg
      from mjlab_myosuite.tasks.tracking.tracking_env_cfg import (
        MyoSuiteTrackingEnvCfg,
      )
      from mjlab_myosuite.wrapper import MyoSuiteVecEnvWrapper
    except ImportError:
      # MyoSuite not available, fall back to original
      return _original_run_play(task_id, cfg)

    # Use the same logic as mjlab's run_play but use gym.make() for MyoSuite
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    DUMMY_MODE = cfg.agent in {"zero", "random"}
    TRAINED_MODE = not DUMMY_MODE

    # Handle MyoSuite configs
    if isinstance(env_cfg, (MyoSuiteEnvCfg, MyoSuiteTrackingEnvCfg)):
      env_cfg.device = device
      if cfg.num_envs is not None:
        env_cfg.num_envs = cfg.num_envs

    # Handle tracking tasks and motion_file
    is_tracking = isinstance(env_cfg, MyoSuiteTrackingEnvCfg)
    if is_tracking:
      if cfg.motion_file is not None:
        if env_cfg.commands is not None and hasattr(env_cfg.commands, "motion"):
          env_cfg.commands.motion.motion_file = cfg.motion_file
      elif cfg.registry_name:
        # Handle wandb registry
        registry_name = cfg.registry_name
        if ":" not in registry_name:
          registry_name = registry_name + ":latest"
        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        if env_cfg.commands is not None and hasattr(env_cfg.commands, "motion"):
          env_cfg.commands.motion.motion_file = str(
            Path(artifact.download()) / "motion.npz"
          )

    # Handle video settings
    if cfg.video_height is not None:
      if hasattr(env_cfg, "viewer") and hasattr(env_cfg.viewer, "height"):
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
      if hasattr(env_cfg, "viewer") and hasattr(env_cfg.viewer, "width"):
        env_cfg.viewer.width = cfg.video_width

    # Create environment using gym.make() instead of ManagerBasedRlEnv
    render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
    env = gym.make(task_id, cfg=env_cfg, device=device, render_mode=render_mode)

    # Handle video recording (same as mjlab)
    if TRAINED_MODE and cfg.video:
      from mjlab.utils.os import get_wandb_checkpoint_path

      log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
      if cfg.checkpoint_file is not None:
        log_dir = Path(cfg.checkpoint_file).parent
      elif cfg.wandb_run_path is not None:
        resume_path, _ = get_wandb_checkpoint_path(
          log_root_path, Path(cfg.wandb_run_path)
        )
        log_dir = resume_path.parent
      else:
        log_dir = log_root_path

      print("[INFO] Recording videos during play")
      env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(log_dir / "videos" / "play"),
        step_trigger=lambda step: step == 0,
        video_length=cfg.video_length,
        disable_logger=True,
      )

    # For MyoSuite tasks, implement the full play logic ourselves using gym.make()
    # instead of ManagerBasedRlEnv, since ManagerBasedRlEnv doesn't work well with
    # MyoSuite configs (it expects mjlab's observation system).

    # Find MyoSuiteVecEnvWrapper and set clip_actions
    def find_myosuite_wrapper(env_obj):
      """Unwrap environment to find MyoSuiteVecEnvWrapper."""
      current = env_obj
      max_depth = 10
      depth = 0
      visited = set()

      while depth < max_depth:
        if isinstance(current, MyoSuiteVecEnvWrapper):
          return current

        obj_id = id(current)
        if obj_id in visited:
          break
        visited.add(obj_id)

        next_env = None
        if hasattr(current, "env"):
          try:
            candidate = current.env
            if candidate is not current and candidate is not None:
              next_env = candidate
          except (AttributeError, TypeError):
            pass

        if next_env is None and hasattr(current, "unwrapped"):
          try:
            candidate = current.unwrapped
            if candidate is not current and candidate is not None:
              next_env = candidate
          except (AttributeError, TypeError):
            pass

        if next_env is None or next_env is current:
          break

        current = next_env
        depth += 1

      return None

    myosuite_wrapper = find_myosuite_wrapper(env)
    if myosuite_wrapper is not None:
      myosuite_wrapper.clip_actions = agent_cfg.clip_actions
      # Update wrapper device to match policy device
      myosuite_wrapper.device = torch.device(device)
      myosuite_wrapper.device_str = device
      env_for_runner = myosuite_wrapper
    else:
      env_for_runner = env

    # Handle dummy mode (zero/random agents)
    if DUMMY_MODE:
      # Get action shape
      action_shape = None
      if hasattr(env_for_runner.unwrapped, "single_action_space"):
        action_shape = getattr(
          env_for_runner.unwrapped.single_action_space, "shape", None
        )
      else:
        action_shape = getattr(env_for_runner.unwrapped.action_space, "shape", None)
        if isinstance(action_shape, tuple) and len(action_shape) > 1:
          if getattr(env_for_runner.unwrapped, "num_envs", 1) == action_shape[0]:
            action_shape = action_shape[1:]

      env_device = (
        env_for_runner.unwrapped.device
        if hasattr(env_for_runner.unwrapped, "device")
        else torch.device("cpu")
      )

      if cfg.agent == "zero":

        class PolicyZero:
          def __call__(self, obs) -> torch.Tensor:
            del obs
            per_env_shape = action_shape if isinstance(action_shape, tuple) else ()
            return torch.zeros(
              (getattr(env_for_runner.unwrapped, "num_envs", 1),) + per_env_shape,
              device=env_device,
            )

        policy = PolicyZero()
      else:

        class PolicyRandom:
          def __call__(self, obs) -> torch.Tensor:
            del obs
            per_env_shape = action_shape if isinstance(action_shape, tuple) else ()
            return (
              2
              * torch.rand(
                (getattr(env_for_runner.unwrapped, "num_envs", 1),) + per_env_shape,
                device=env_device,
              )
              - 1
            )

        policy = PolicyRandom()
    else:
      # TRAINED_MODE: Load policy from checkpoint
      from dataclasses import asdict
      from typing import Any, cast

      from mjlab.utils.os import get_wandb_checkpoint_path
      from rsl_rl.runners.on_policy_runner import OnPolicyRunner

      from mjlab_myosuite.tasks.tracking.rl import (
        MyoSuiteMotionTrackingOnPolicyRunner,
      )

      log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
      if cfg.checkpoint_file is not None:
        resume_path = Path(cfg.checkpoint_file)
        if not resume_path.exists():
          raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        print(f"[INFO]: Loading checkpoint: {resume_path.name}")
      else:
        if cfg.wandb_run_path is None:
          raise ValueError(
            "`wandb_run_path` is required when `checkpoint_file` is not provided."
          )
        resume_path, was_cached = get_wandb_checkpoint_path(
          log_root_path, Path(cfg.wandb_run_path)
        )
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
        )

      # Create runner based on task type
      if is_tracking:
        runner = MyoSuiteMotionTrackingOnPolicyRunner(
          cast(Any, env_for_runner),
          asdict(agent_cfg),
          log_dir=str(resume_path.parent),
          device=device,
          registry_name=None,
        )
      else:
        runner = OnPolicyRunner(
          cast(Any, env_for_runner),
          asdict(agent_cfg),
          log_dir=str(resume_path.parent),
          device=device,
        )
      runner.load(str(resume_path), map_location=device)
      policy = runner.get_inference_policy(device=device)

    # Handle viewer
    from typing import cast

    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
    from mjlab.viewer.base import EnvProtocol

    resolved_viewer = cfg.viewer
    if resolved_viewer == "auto":
      # Auto-detect: prefer viser if available, fallback to native
      if ViserPlayViewer is not None:
        resolved_viewer = "viser"
      elif NativeMujocoViewer is not None:
        resolved_viewer = "native"
      else:
        raise RuntimeError("No viewer available")

    env_for_viewer = env_for_runner

    # Ensure forward kinematics are computed before viewer starts
    if hasattr(env_for_viewer, "sim"):
      import mujoco

      sim = env_for_viewer.sim
      if hasattr(sim, "_env"):
        env_obj = sim._env
        mj_model = getattr(env_obj, "mj_model", getattr(env_obj, "model", None))
        mj_data = getattr(env_obj, "mj_data", getattr(env_obj, "data", None))
        if mj_model is not None and mj_data is not None:
          mujoco.mj_forward(mj_model, mj_data)

    if resolved_viewer == "native":
      if NativeMujocoViewer is None:
        raise ImportError("NativeMujocoViewer not available in this mjlab version")
      if EnvProtocol is not None:
        NativeMujocoViewer(cast(EnvProtocol, env_for_viewer), policy).run()  # type: ignore[arg-type]
      else:
        NativeMujocoViewer(env_for_viewer, policy).run()  # type: ignore[arg-type]
    elif resolved_viewer == "viser":
      if ViserPlayViewer is None:
        raise ImportError(
          "ViserPlayViewer not available. Install viser or use --viewer native"
        )

      # Patch ViserMujocoScene._add_fixed_geometry to handle textured planes
      try:
        import mujoco
        from mjlab.viewer.viser.scene import ViserMujocoScene
        from mujoco import mj_id2name, mjtGeom, mjtObj

        from mjlab_myosuite.viewer_helpers import (
          create_textured_plane,
          find_textured_geometries,
        )

        original_add_fixed_geometry = ViserMujocoScene._add_fixed_geometry

        def patched_add_fixed_geometry(self):
          """Patched version that adds textured planes as meshes instead of grids."""
          # Find all textured geometries
          textured_geom_info = find_textured_geometries(self.mj_model)

          body_geoms_visual: dict[int, list[int]] = {}
          body_geoms_collision: dict[int, list[int]] = {}

          for i in range(self.mj_model.ngeom):
            body_id = self.mj_model.geom_bodyid[i]
            target = (
              body_geoms_collision if self._is_collision_geom(i) else body_geoms_visual
            )
            target.setdefault(body_id, []).append(i)

          # Process all bodies with geoms.
          all_bodies = set(body_geoms_visual.keys()) | set(body_geoms_collision.keys())

          for body_id in all_bodies:
            # Get body name.
            from mjlab.viewer.viser.conversions import (
              get_body_name,
              is_fixed_body,
              merge_geoms,
            )

            body_name = get_body_name(self.mj_model, body_id)

            # Fixed world geometry. We'll assume this is shared between all environments.
            if is_fixed_body(self.mj_model, body_id):
              # Create both visual and collision geoms for fixed bodies (terrain, floor, etc.)
              # but show them all since they're static.
              all_geoms = []
              if body_id in body_geoms_visual:
                all_geoms.extend(body_geoms_visual[body_id])
              if body_id in body_geoms_collision:
                all_geoms.extend(body_geoms_collision[body_id])

              if not all_geoms:
                continue

              # Iterate over geoms - handle each individually to preserve textures
              from mjlab.viewer.viser.conversions import (
                mujoco_mesh_to_trimesh,
              )

              nonplane_geom_ids: list[int] = []
              for geom_id in all_geoms:
                geom_type = self.mj_model.geom_type[geom_id]
                matid, texid = textured_geom_info.get(geom_id, (-1, -1))

                # Check if this is a mesh geometry (actual mesh data)
                mesh_id = self.mj_model.geom_dataid[geom_id]
                if mesh_id >= 0 and self.mj_model.mesh_vertnum[mesh_id] > 0:
                  # Mesh geometry - use mujoco_mesh_to_trimesh which preserves textures
                  mesh = mujoco_mesh_to_trimesh(self.mj_model, geom_id, verbose=False)
                  # DO NOT TOUCH mesh.visual - MuJoCo already gave us UVs + textures
                  geom_name = (
                    mj_id2name(
                      self.mj_model,
                      mjtObj.mjOBJ_GEOM,
                      geom_id,
                    )
                    or f"geom_{geom_id}"
                  )
                  # Apply geom transform (not body transform for individual geoms)
                  self.server.scene.add_mesh_trimesh(
                    f"/fixed_bodies/{body_name}/{geom_name}",
                    mesh,
                    cast_shadow=False,
                    receive_shadow=0.2,
                    position=self.mj_model.geom_pos[geom_id]
                    + self.mj_model.body_pos[body_id],
                    wxyz=self.mj_model.geom_quat[geom_id],
                    visible=True,
                  )
                  continue  # Skip adding to nonplane_geom_ids

                # Check if this is a BOX geom or other primitive (with texture support)
                if geom_type == mjtGeom.mjGEOM_BOX or (
                  geom_type not in (mjtGeom.mjGEOM_MESH, mjtGeom.mjGEOM_PLANE)
                  and mesh_id < 0
                ):
                  from mjlab_myosuite.viewer_helpers import (
                    create_primitive_mesh,
                  )

                  # Create the primitive mesh with texture support
                  mesh = create_primitive_mesh(self.mj_model, geom_id, matid, texid)

                  geom_name = (
                    mj_id2name(
                      self.mj_model,
                      mjtObj.mjOBJ_GEOM,
                      geom_id,
                    )
                    or f"geom_{geom_id}"
                  )
                  print(
                    f"{body_name} -- {geom_name} -- {self.mj_model.geom_pos[geom_id]} -- {self.mj_model.body_pos[body_id]}"
                  )
                  self.server.scene.add_mesh_trimesh(
                    f"/fixed_bodies/{body_name}/{geom_name}",
                    mesh,
                    cast_shadow=False,
                    receive_shadow=0.2,
                    position=self.mj_model.geom_pos[geom_id]
                    + self.mj_model.body_pos[body_id],
                    wxyz=self.mj_model.geom_quat[geom_id],
                    visible=True,
                  )
                  continue  # Skip adding to nonplane_geom_ids

                # Check if this is a plane with a texture
                if geom_type == mjtGeom.mjGEOM_PLANE:
                  if texid >= 0:
                    # Textured plane - add as mesh instead of grid

                    mesh = create_textured_plane(self.mj_model, geom_id, matid, texid)
                    if mesh is not None:
                      geom_name = (
                        mj_id2name(
                          self.mj_model,
                          mjtObj.mjOBJ_GEOM,
                          geom_id,
                        )
                        or f"geom_{geom_id}"
                      )
                      self.server.scene.add_mesh_trimesh(
                        f"/fixed_bodies/{body_name}/{geom_name}",
                        mesh,
                        cast_shadow=False,
                        receive_shadow=0.2,
                        position=self.mj_model.geom_pos[geom_id]
                        + self.mj_model.body_pos[body_id],
                        wxyz=self.mj_model.geom_quat[geom_id],
                        visible=True,
                      )
                      continue  # Skip adding as grid

                  # Untextured plane - add as infinite grid (original behavior)
                  geom_name = (
                    mj_id2name(self.mj_model, mjtObj.mjOBJ_GEOM, geom_id)
                    or f"geom_{geom_id}"
                  )
                  self.server.scene.add_grid(
                    f"/fixed_bodies/{body_name}/{geom_name}",
                    width=2000.0,
                    height=2000.0,
                    infinite_grid=True,
                    fade_distance=50.0,
                    shadow_opacity=0.2,
                    position=self.mj_model.geom_pos[geom_id]
                    + self.mj_model.body_pos[body_id],
                    wxyz=self.mj_model.geom_quat[geom_id],
                  )
                else:
                  # Other primitive types - add to list for merging
                  nonplane_geom_ids.append(geom_id)

              # Handle remaining non-plane, non-mesh geoms by merging (original behavior)
              if len(nonplane_geom_ids) > 0:
                self.server.scene.add_mesh_trimesh(
                  f"/fixed_bodies/{body_name}",
                  merge_geoms(self.mj_model, nonplane_geom_ids),
                  cast_shadow=False,
                  receive_shadow=0.2,
                  position=self.mj_model.body(body_id).pos,
                  wxyz=self.mj_model.body(body_id).quat,
                  visible=True,
                )

        # Apply patch
        ViserMujocoScene._add_fixed_geometry = patched_add_fixed_geometry

        try:
          if EnvProtocol is not None:
            ViserPlayViewer(cast(EnvProtocol, env_for_viewer), policy).run()  # type: ignore[arg-type]
          else:
            ViserPlayViewer(env_for_viewer, policy).run()  # type: ignore[arg-type]
        finally:
          # Restore original method
          ViserMujocoScene._add_fixed_geometry = original_add_fixed_geometry
      except Exception as e:
        # If patching fails, fall back to original behavior
        import warnings

        warnings.warn(
          f"Failed to patch ViserMujocoScene for textured planes: {e}",
          UserWarning,
          stacklevel=2,
        )
        if EnvProtocol is not None:
          ViserPlayViewer(cast(EnvProtocol, env_for_viewer), policy).run()  # type: ignore[arg-type]
        else:
          ViserPlayViewer(env_for_viewer, policy).run()  # type: ignore[arg-type]
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

    env.close()
    return
  else:
    # Not a MyoSuite task, use original run_play
    return _original_run_play(task_id, cfg)


# Patch mjlab's run_play with our version
mjlab_play_module.run_play = _patched_run_play

# Now import and run mjlab's native play script
# This import happens AFTER registration and patching
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.scripts.play import main as mjlab_main

# Re-export for backward compatibility with tests and other code
# Note: mjlab's PlayConfig already has motion_file, so we can use it directly
main = mjlab_main
__all__ = ["main", "PlayConfig", "run_play"]

if __name__ == "__main__":
  main()
