"""Script to train RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import tyro

# Import mjlab_myosuite to trigger auto-registration of MyoSuite environments
# This MUST happen before tyro import and before any mjlab imports
# to ensure registration completes before argument parsing
try:
  import mjlab_myosuite  # noqa: F401

  # Force registration to complete by accessing the registry
  _ = list(gym.registry.keys())  # Trigger any lazy registration
except ImportError:
  pass  # MyoSuite not available, skip registration

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

# Optional imports for tracking tasks
try:
  from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
  from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
except ImportError:
  MotionTrackingOnPolicyRunner = None  # type: ignore
  TrackingEnvCfg = None  # type: ignore

try:
  from mjlab_myosuite.config import MyoSuiteEnvCfg
  from mjlab_myosuite.rl import MyoSuiteOnPolicyRunner
  from mjlab_myosuite.wrapper import MyoSuiteVecEnvWrapper
except ImportError:
  # MyoSuite not available
  MyoSuiteEnvCfg = None  # type: ignore
  MyoSuiteOnPolicyRunner = None  # type: ignore
  MyoSuiteVecEnvWrapper = None  # type: ignore
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


from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class TrainConfig:
  env: Any
  agent: RslRlOnPolicyRunnerCfg
  registry_name: str | None = None
  device: str = "cuda:0"
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False


def run_train(task: str, cfg: TrainConfig) -> None:
  configure_torch_backends()

  registry_name: str | None = None

  if TrackingEnvCfg is not None and isinstance(cfg.env, TrackingEnvCfg):
    if not cfg.registry_name:
      raise ValueError("Must provide --registry-name for tracking tasks.")

    # Check if the registry name includes alias, if not, append ":latest".
    registry_name = cast(str, cfg.registry_name)
    if ":" not in registry_name:
      registry_name = registry_name + ":latest"
    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    cfg.env.commands.motion.motion_file = str(Path(artifact.download()) / "motion.npz")

  # Enable NaN guard if requested
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  # Specify directory for logging experiments.
  log_root_path = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
  log_root_path.resolve()
  print(f"[INFO] Logging experiment in directory: {log_root_path}")
  log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if cfg.agent.run_name:
    log_dir += f"_{cfg.agent.run_name}"
  log_dir = log_root_path / log_dir

  # Check if this is a MyoSuite environment
  is_myosuite = task.startswith("Mjlab-MyoSuite") or (
    MyoSuiteEnvCfg is not None and isinstance(cfg.env, MyoSuiteEnvCfg)
  )

  # For MyoSuite environments, ensure device is set in config to match training device
  # This ensures observations are on the correct device
  if is_myosuite and MyoSuiteEnvCfg is not None and isinstance(cfg.env, MyoSuiteEnvCfg):
    cfg.env.device = cfg.device

  env = gym.make(
    task, cfg=cfg.env, device=cfg.device, render_mode="rgb_array" if cfg.video else None
  )

  resume_path = (
    get_checkpoint_path(log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint)
    if cfg.agent.resume
    else None
  )

  # Helper function to find MyoSuiteVecEnvWrapper by unwrapping (needed for MyoSuite)
  def find_myosuite_wrapper(env_obj):
    """Unwrap environment to find MyoSuiteVecEnvWrapper."""
    current = env_obj
    max_depth = 10
    depth = 0
    visited = set()  # Prevent infinite loops

    while depth < max_depth:
      if (MyoSuiteVecEnvWrapper is not None) and isinstance(
        current, MyoSuiteVecEnvWrapper
      ):
        return current

      # Prevent infinite loops
      obj_id = id(current)
      if obj_id in visited:
        break
      visited.add(obj_id)

      # Try to unwrap - check multiple possible attribute names
      next_env = None

      # Try .env attribute (most common in gymnasium wrappers like OrderEnforcing)
      if hasattr(current, "env"):
        try:
          candidate = current.env
          if candidate is not current and candidate is not None:
            next_env = candidate
        except (AttributeError, TypeError):
          pass

      # Try .unwrapped property
      if next_env is None and hasattr(current, "unwrapped"):
        try:
          candidate = current.unwrapped
          if candidate is not current and candidate is not None:
            next_env = candidate
        except (AttributeError, TypeError):
          pass

      # Try ._env (private attribute some wrappers use)
      if next_env is None and hasattr(current, "_env"):
        try:
          candidate = current._env
          if candidate is not current and candidate is not None:
            next_env = candidate
        except (AttributeError, TypeError):
          pass

      if next_env is None or next_env is current:
        # Can't unwrap further
        break

      current = next_env
      depth += 1

    return None

  # Find the MyoSuiteVecEnvWrapper before wrapping with RecordVideo (if MyoSuite)
  myosuite_wrapper = find_myosuite_wrapper(env) if is_myosuite else None

  if cfg.video:
    env = gym.wrappers.RecordVideo(
      env,
      video_folder=os.path.join(log_dir, "videos", "train"),
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")
    # Re-find the wrapper after RecordVideo wraps it
    if is_myosuite and myosuite_wrapper is None:
      myosuite_wrapper = find_myosuite_wrapper(env)

  if is_myosuite:
    # Unwrap gymnasium's OrderEnforcing wrapper to get to our MyoSuiteVecEnvWrapper
    # The wrapper chain might be: OrderEnforcing -> RecordVideo -> MyoSuiteVecEnvWrapper
    # We need the unwrapped env because OrderEnforcing doesn't forward get_observations()

    if myosuite_wrapper is not None:
      # Use the unwrapped wrapper directly
      env = myosuite_wrapper
      env.clip_actions = cfg.agent.clip_actions
    else:
      # If we couldn't find it, try one more time with current env
      myosuite_wrapper = find_myosuite_wrapper(env)
      if myosuite_wrapper is not None:
        env = myosuite_wrapper
        env.clip_actions = cfg.agent.clip_actions
      else:
        # If still not found, raise an error with helpful info
        raise RuntimeError(
          "Could not find MyoSuiteVecEnvWrapper in environment wrapper chain. "
          f"Environment type: {type(env)}, "
          f"has .env: {hasattr(env, 'env')}, "
          f"has .unwrapped: {hasattr(env, 'unwrapped')}, "
          f"unwrapped type: {type(getattr(env, 'unwrapped', None))}"
        )
  else:
    # Standard mjlab environment - wrap with RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)  # type: ignore[arg-type]

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  if TrackingEnvCfg is not None and isinstance(cfg.env, TrackingEnvCfg):
    if MotionTrackingOnPolicyRunner is None:
      raise ImportError("MotionTrackingOnPolicyRunner not available")
    runner = MotionTrackingOnPolicyRunner(
      env, agent_cfg, str(log_dir), cfg.device, registry_name
    )
  elif is_myosuite:
    # MyoSuite environments use a custom runner that skips ONNX export
    assert MyoSuiteOnPolicyRunner is not None
    runner = MyoSuiteOnPolicyRunner(env, agent_cfg, str(log_dir), cfg.device)
  else:
    runner = VelocityOnPolicyRunner(env, agent_cfg, str(log_dir), cfg.device)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
  dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def main():
  # Parse first argument manually to avoid tyro evaluating choices before registration
  if len(sys.argv) < 2:
    print("Usage: python scripts/train.py <task-id> [options...]")
    print(
      "Example: python scripts/train.py Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 --agent.max-iterations 2000"
    )
    sys.exit(1)

  # Get the task ID from command line (before tyro processes it)
  chosen_task = sys.argv[1]
  remaining_args = sys.argv[2:]

  # Ensure MyoSuite environments are registered
  task_prefix = "Mjlab-"
  if chosen_task.startswith(task_prefix):
    # Force registration - import happens at module level but ensure it completed
    try:
      import mjlab_myosuite  # noqa: F401

      # Force a registry access to ensure registration completed
      _ = list(gym.registry.keys())
      # Small delay to ensure async operations complete
      import time

      time.sleep(0.1)
    except ImportError:
      pass  # MyoSuite not available, skip registration

    # Verify the task exists - try multiple times in case of timing issues
    for _ in range(3):
      if chosen_task in gym.registry:
        break
      # Re-trigger registration
      try:
        from mjlab_myosuite.registration import register_myosuite_envs

        register_myosuite_envs()
      except Exception:
        pass
      import time

      time.sleep(0.1)

    if chosen_task not in gym.registry:
      available_tasks = [k for k in gym.registry.keys() if k.startswith(task_prefix)]
      print(
        f"[ERROR] Task '{chosen_task}' not found in registry.\n"
        f"Found {len(available_tasks)} tasks with prefix '{task_prefix}'."
      )
      if available_tasks:
        # Show similar tasks
        task_suffix = chosen_task.split("-")[-1] if "-" in chosen_task else chosen_task
        similar = [t for t in available_tasks if task_suffix in t]
        if similar:
          print(f"Similar tasks: {similar[:5]}")
        else:
          print(f"Sample tasks: {available_tasks[:5]}")
      else:
        print(
          "[INFO] No MyoSuite tasks found. "
          "Make sure MyoSuite is installed and mjlab_myosuite imported successfully."
        )
      sys.exit(1)

  del task_prefix

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  env_cfg = load_cfg_from_registry(chosen_task, "env_cfg_entry_point")
  agent_cfg = load_cfg_from_registry(chosen_task, "rl_cfg_entry_point")
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig(env=env_cfg, agent=agent_cfg),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del env_cfg, agent_cfg, remaining_args

  run_train(chosen_task, args)


if __name__ == "__main__":
  main()
