"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional, cast

import gymnasium as gym
import torch
import tyro

# Import mjlab_myosuite to trigger auto-registration of MyoSuite environments
try:
  import mjlab_myosuite  # noqa: F401
except ImportError:
  pass  # MyoSuite not available, skip registration

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Optional imports for tracking tasks
try:
  from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
  from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
except ImportError:
  MotionTrackingOnPolicyRunner = None  # type: ignore
  TrackingEnvCfg = None  # type: ignore
# Try to import cfg loading utilities
try:
  from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
    load_cfg_from_registry,
  )
except ImportError:
  # Fallback: implement a simple version for gymnasium registry
  # Match the signature of the imported function (uses task_name parameter)
  def load_cfg_from_registry(task_name: str, entry_point_key: str) -> Any:  # type: ignore[no-redef]
    """Load configuration from gymnasium registry entry point.

    Args:
      task_name: Task/Environment ID
      entry_point_key: Key in kwargs (e.g., 'env_cfg_entry_point' or 'rl_cfg_entry_point')

    Returns:
      The configuration object
    """
    if task_name not in gym.registry:
      raise KeyError(f"Environment {task_name} not found in registry")

    spec = gym.registry[task_name]
    if spec.kwargs is None or entry_point_key not in spec.kwargs:
      raise KeyError(
        f"Entry point '{entry_point_key}' not found for environment {task_name}"
      )

    entry_point_str = spec.kwargs[entry_point_key]
    # Parse entry point string like "module.path:function"
    if ":" in entry_point_str:
      module_path, func_name = entry_point_str.rsplit(":", 1)
    else:
      module_path, func_name = entry_point_str.rsplit(".", 1)

    # Import and call the function
    module = __import__(module_path, fromlist=[func_name])
    func = getattr(module, func_name)
    return func()


from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

# Optional viewer imports - different mjlab versions may have different viewers
try:
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
  from mjlab.viewer.base import EnvProtocol

  # Alias for backward compatibility
  ViserViewer = ViserPlayViewer  # type: ignore[assignment]
except ImportError:
  # Try alternative import paths
  try:
    from mjlab.viewer import NativeMujocoViewer

    ViserPlayViewer = None  # type: ignore
    ViserViewer = None  # type: ignore
    from mjlab.viewer.base import EnvProtocol
  except ImportError:
    # Viewers not available
    NativeMujocoViewer = None  # type: ignore
    ViserPlayViewer = None  # type: ignore
    ViserViewer = None  # type: ignore
    EnvProtocol = None  # type: ignore

try:
  from mjlab_myosuite.config import MyoSuiteEnvCfg
  from mjlab_myosuite.wrapper import MyoSuiteVecEnvWrapper
except ImportError:
  MyoSuiteEnvCfg = None  # type: ignore
  MyoSuiteVecEnvWrapper = None  # type: ignore

ViewerChoice = Literal["auto", "native", "viser"]
ResolvedViewer = Literal["native", "viser"]


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: ViewerChoice = "auto"
  # Viser-specific options
  viser_port: int | None = None
  """Port for Viser web server (default: auto-assigned)"""
  viser_host: str = "localhost"
  """Host for Viser web server"""


def _resolve_viewer_choice(choice: ViewerChoice) -> ResolvedViewer:
  """Resolve viewer choice, falling back to available viewers if needed."""
  if choice != "auto":
    resolved = cast(ResolvedViewer, choice)
    # Check if the requested viewer is available
    if resolved == "viser" and (ViserViewer is None or ViserPlayViewer is None):
      if NativeMujocoViewer is not None:
        print("[WARN]: ViserViewer not available, falling back to native viewer")
        return "native"
      else:
        raise ImportError(
          "Neither ViserViewer nor NativeMujocoViewer is available. "
          "Cannot proceed with visualization."
        )
    if resolved == "native" and NativeMujocoViewer is None:
      if ViserViewer is not None or ViserPlayViewer is not None:
        print("[WARN]: NativeMujocoViewer not available, falling back to viser viewer")
        return "viser"
      else:
        raise ImportError(
          "Neither NativeMujocoViewer nor ViserViewer is available. "
          "Cannot proceed with visualization."
        )
    return resolved

  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  # Prefer viser when no display, but fall back to native if viser is not available
  if has_display:
    # When display is available, prefer native viewer
    if NativeMujocoViewer is not None:
      resolved: ResolvedViewer = "native"
    elif ViserViewer is not None or ViserPlayViewer is not None:
      resolved: ResolvedViewer = "viser"
    else:
      raise ImportError("No viewers available")
  else:
    # When no display, prefer viser (web-based)
    if ViserViewer is not None or ViserPlayViewer is not None:
      resolved: ResolvedViewer = "viser"
    elif NativeMujocoViewer is not None:
      resolved: ResolvedViewer = "native"
    else:
      raise ImportError("No viewers available")
  print(f"[INFO]: Auto-selected viewer: {resolved} (display detected: {has_display})")
  return resolved


def playback_with_viser(
  env: Any,
  policy: Any,
  num_steps: int | None = None,
  reset_on_done: bool = True,
  verbose: bool = True,
  port: int | None = None,
  host: str = "localhost",
) -> None:
  """Playback policy with Viser viewer.

  This utility function provides a convenient way to visualize policy execution
  using the Viser web-based viewer. It handles environment stepping, episode
  resets, and visualization updates.

  Args:
    env: The environment to run (should be compatible with ViserViewer)
    policy: The policy to execute (callable that takes observations and returns actions)
    num_steps: Maximum number of steps to run. If None, runs until interrupted.
    reset_on_done: Whether to automatically reset the environment when episodes end.
    verbose: Whether to print status messages.
    port: Port for Viser web server. If None, uses default/auto-assigned port.
    host: Host for Viser web server. Defaults to "localhost".

  Raises:
    ImportError: If ViserViewer is not available.
    RuntimeError: If the environment is not compatible with ViserViewer.

  Example:
    ```python
    from scripts.play import playback_with_viser
    import gymnasium as gym

    # Create environment and load policy
    env = gym.make("Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0")
    policy = load_policy("path/to/checkpoint.pt")

    # Playback with Viser
    playback_with_viser(env, policy, verbose=True)
    ```
  """
  # Check for ViserPlayViewer (new name) or ViserViewer (old name/alias)
  viser_viewer = ViserPlayViewer if ViserPlayViewer is not None else ViserViewer
  if viser_viewer is None:
    raise ImportError(
      "ViserViewer (ViserPlayViewer) is not available. "
      "Install viser or use a different viewer backend (e.g., --viewer native)."
    )

  if verbose:
    print("[INFO] Starting Viser playback...")
    if num_steps is not None:
      print(f"[INFO] Will run for {num_steps} steps")
    print(f"[INFO] Viser server will be available at http://{host}:{port or 'auto'}")
    print("[INFO] Open the URL shown below in your browser to view the simulation")
    print("[INFO] Press Ctrl+C to stop playback")

  # Ensure forward kinematics are computed for MyoSuite environments
  if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "sim"):
    import mujoco

    sim = env.unwrapped.sim
    if hasattr(sim, "_env"):
      env_obj = sim._env
      mj_model = getattr(env_obj, "mj_model", getattr(env_obj, "model", None))
      mj_data = getattr(env_obj, "mj_data", getattr(env_obj, "data", None))
      if mj_model is not None and mj_data is not None:
        mujoco.mj_forward(mj_model, mj_data)

  # Create and run the Viser viewer
  try:
    # Check if ViserPlayViewer accepts port/host parameters
    viewer_kwargs: dict[str, Any] = {}
    try:
      import inspect

      sig = inspect.signature(viser_viewer.__init__)
      if "port" in sig.parameters:
        viewer_kwargs["port"] = port
      if "host" in sig.parameters:
        viewer_kwargs["host"] = host
    except Exception:
      pass  # If signature inspection fails, just use defaults

    if EnvProtocol is not None:
      viewer = viser_viewer(cast(EnvProtocol, env), policy, **viewer_kwargs)  # type: ignore[arg-type]
    else:
      viewer = viser_viewer(env, policy, **viewer_kwargs)  # type: ignore[arg-type]

    # The viewer.run() method handles the main loop
    # If we need custom stepping logic, we can extend this
    viewer.run()
  except KeyboardInterrupt:
    if verbose:
      print("\n[INFO] Playback interrupted by user")
  except Exception as e:
    if verbose:
      print(f"[ERROR] Playback failed: {e}")
    raise


def run_play(task: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Check if this is a MyoSuite environment
  is_myosuite = task.startswith("Mjlab-MyoSuite")

  env_cfg = cast(
    ManagerBasedRlEnvCfg, load_cfg_from_registry(task, "env_cfg_entry_point")
  )
  agent_cfg = cast(
    RslRlOnPolicyRunnerCfg, load_cfg_from_registry(task, "rl_cfg_entry_point")
  )

  # For MyoSuite environments, set device in config to match policy device
  # This ensures observations are on the correct device
  if (
    is_myosuite and (MyoSuiteEnvCfg is not None) and isinstance(env_cfg, MyoSuiteEnvCfg)
  ):
    env_cfg.device = device

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  if TrackingEnvCfg is not None and isinstance(env_cfg, TrackingEnvCfg):
    if DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require `registry_name` when using dummy agents."
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cast(str, cfg.registry_name)
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      env_cfg.commands.motion.motion_file = str(
        Path(artifact.download()) / "motion.npz"
      )
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        env_cfg.commands.motion.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          env_cfg.commands.motion.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Optional[Path] = None
  resume_path: Optional[Path] = None
  if TRAINED_MODE:
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
      # Extract run_id and checkpoint name from path for display
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )

  env = gym.make(task, cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    env = gym.wrappers.RecordVideo(
      env,
      video_folder=str(Path(log_dir) / "videos" / "play"),  # type: ignore[arg-type]
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  # Handle MyoSuite environments differently
  if is_myosuite:
    # Helper function to find MyoSuiteVecEnvWrapper by unwrapping
    def find_myosuite_wrapper(env_obj):
      """Unwrap environment to find MyoSuiteVecEnvWrapper."""
      current = env_obj
      max_depth = 10
      depth = 0
      visited = set()

      while depth < max_depth:
        if (MyoSuiteVecEnvWrapper is not None) and isinstance(
          current, MyoSuiteVecEnvWrapper
        ):
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

    # Find the MyoSuiteVecEnvWrapper
    myosuite_wrapper = find_myosuite_wrapper(env)
    if myosuite_wrapper is not None:
      env = myosuite_wrapper
      env.clip_actions = agent_cfg.clip_actions
      # CRITICAL: Update wrapper device to match policy device for inference
      # This ensures observations are on the same device as the policy
      env.device = torch.device(device)
      env.device_str = device
    else:
      raise RuntimeError(
        "Could not find MyoSuiteVecEnvWrapper in environment wrapper chain."
      )
  else:
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  if DUMMY_MODE:
    # Get action shape - for vectorized environments, use single_action_space
    action_shape = None
    if hasattr(env.unwrapped, "single_action_space"):
      action_shape = getattr(env.unwrapped.single_action_space, "shape", None)  # type: ignore
    else:
      action_shape = getattr(env.unwrapped.action_space, "shape", None)  # type: ignore
      # For vectorized spaces, remove batch dimension if present
      if (
        isinstance(action_shape, tuple)
        and len(action_shape) > 1
        and getattr(env.unwrapped, "num_envs", 1) == action_shape[0]
      ):
        action_shape = action_shape[1:]

    # Get device
    env_device = (
      env.unwrapped.device if hasattr(env.unwrapped, "device") else torch.device("cpu")
    )

    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          # Return actions with shape (num_envs, action_dim)
          per_env_shape = action_shape if isinstance(action_shape, tuple) else ()
          return torch.zeros(
            (getattr(env.unwrapped, "num_envs", 1),) + per_env_shape, device=env_device
          )

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          # Return actions with shape (num_envs, action_dim)
          per_env_shape = action_shape if isinstance(action_shape, tuple) else ()
          return (
            2
            * torch.rand(
              (getattr(env.unwrapped, "num_envs", 1),) + per_env_shape,
              device=env_device,
            )
            - 1
          )

      policy = PolicyRandom()
  else:
    if TrackingEnvCfg is not None and isinstance(env_cfg, TrackingEnvCfg):
      if MotionTrackingOnPolicyRunner is None:
        raise ImportError("MotionTrackingOnPolicyRunner not available")
      runner = MotionTrackingOnPolicyRunner(
        env, asdict(agent_cfg), log_dir=str(log_dir), device=device
      )
    else:
      runner = OnPolicyRunner(
        env, asdict(agent_cfg), log_dir=str(log_dir), device=device
      )
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

  resolved_viewer = _resolve_viewer_choice(cfg.viewer)

  # For MyoSuite environments, ensure forward kinematics are computed before viewer starts
  if is_myosuite and hasattr(env.unwrapped, "sim"):
    import mujoco

    sim = env.unwrapped.sim
    if hasattr(sim, "_env"):
      # Ensure forward kinematics are computed for initial visualization
      # Support both standard and mjx/warp versions
      env_obj = sim._env
      mj_model = getattr(env_obj, "mj_model", getattr(env_obj, "model", None))
      mj_data = getattr(env_obj, "mj_data", getattr(env_obj, "data", None))
      if mj_model is not None and mj_data is not None:
        mujoco.mj_forward(mj_model, mj_data)

  if resolved_viewer == "native":
    if NativeMujocoViewer is None:
      raise ImportError("NativeMujocoViewer not available in this mjlab version")
    if EnvProtocol is not None:
      NativeMujocoViewer(cast(EnvProtocol, env), policy).run()  # type: ignore[arg-type]
    else:
      NativeMujocoViewer(env, policy).run()  # type: ignore[arg-type]
  elif resolved_viewer == "viser":
    # Use the dedicated playback utility for better error handling and features
    try:
      playback_with_viser(
        env,
        policy,
        verbose=True,
        port=cfg.viser_port,
        host=cfg.viser_host,
      )
    except ImportError as e:
      # If ViserViewer is not available, provide helpful error message
      print(f"[ERROR] {e}")
      print("[INFO] Try using --viewer native instead, or install viser package.")
      raise
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument manually to avoid tyro evaluating choices before registration
  if len(sys.argv) < 2:
    print("Usage: python scripts/play.py <task-id> [options...]")
    print(
      "Example: python scripts/play.py Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 --checkpoint_file <path>"
    )
    sys.exit(1)

  # Get the task ID from command line (before tyro processes it)
  chosen_task = sys.argv[1]
  remaining_args = sys.argv[2:]

  # Ensure MyoSuite environments are registered
  task_prefix = "Mjlab-"
  if chosen_task.startswith(task_prefix):
    try:
      import mjlab_myosuite  # noqa: F401
    except ImportError:
      pass  # MyoSuite not available, skip registration

    # Verify the task exists
    if chosen_task not in gym.registry:
      available_tasks = [k for k in gym.registry.keys() if k.startswith(task_prefix)]
      print(
        f"[ERROR] Task '{chosen_task}' not found in registry.\n"
        f"Found {len(available_tasks)} tasks with prefix '{task_prefix}'."
      )
      if available_tasks:
        # Show similar tasks
        similar = [t for t in available_tasks if chosen_task.split("-")[-1] in t]
        if similar:
          print(f"Similar tasks: {similar[:5]}")
        else:
          print(f"Sample tasks: {available_tasks[:5]}")
      sys.exit(1)

  del task_prefix

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  env_cfg = load_cfg_from_registry(chosen_task, "env_cfg_entry_point")
  agent_cfg = load_cfg_from_registry(chosen_task, "rl_cfg_entry_point")
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del env_cfg, agent_cfg, remaining_args

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()


# Export the playback utility for use in other scripts
__all__ = ["playback_with_viser", "run_play", "PlayConfig"]
