"""Wrapper script for mjlab native train script with MyoSuite environment registration.

This script ensures MyoSuite environments are registered before mjlab's native
train script runs, allowing MyoSuite tasks to be used with mjlab's native CLI.

For MyoSuite tasks, this script patches mjlab's run_train to use ManagerBasedRlEnv
(with MyoSuite wrapper), inheriting all other logic from mjlab.
"""

# Import mjlab_myosuite FIRST to trigger auto-registration of MyoSuite environments
# This MUST happen before any mjlab imports to ensure registration completes
# before tyro evaluates choices
import time
from dataclasses import dataclass

from mjlab.scripts.train import TrainConfig as MjlabTrainConfig
from mjlab.scripts.train import main as mjlab_main

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

# Import play.py to trigger ManagerBasedRlEnv patch
# This ensures _patched_manager_init is available when ManagerBasedRlEnv is created
try:
  import mjlab_myosuite.scripts.play  # noqa: F401
except ImportError:
  pass  # play.py not available, patches won't work

# Now import mjlab's train module and patch run_train for MyoSuite tasks
from pathlib import Path

from mjlab.scripts import train as mjlab_train_module

# Store the original run_train function
_original_run_train = mjlab_train_module.run_train


def _patched_run_train(task_id: str, cfg, log_dir: Path) -> None:
  """Patched run_train that uses ManagerBasedRlEnv for MyoSuite tasks.

  This function inherits all logic from mjlab's original run_train, but
  uses ManagerBasedRlEnv (with MyoSuite wrapper) for MyoSuite tasks instead
  of directly using gym.make(). The ManagerBasedRlEnv is patched to wrap
  the MyoSuite environment.
  """
  # Check if this is a MyoSuite task
  if task_id.startswith("Mjlab-MyoSuite"):
    # Import MyoSuite-specific components
    try:
      from mjlab_myosuite.config import MyoSuiteEnvCfg
      from mjlab_myosuite.rl.runner import MyoSuiteOnPolicyRunner
      from mjlab_myosuite.tasks.tracking.rl import (
        MyoSuiteMotionTrackingOnPolicyRunner,
      )
      from mjlab_myosuite.tasks.tracking.tracking_env_cfg import (
        MyoSuiteTrackingEnvCfg,
      )
      from mjlab_myosuite.wrapper import MyoSuiteVecEnvWrapper
    except ImportError:
      # MyoSuite not available, fall back to original
      return _original_run_train(task_id, cfg, log_dir)

    # Import mjlab utilities we need
    import os
    from dataclasses import asdict

    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.os import dump_yaml, get_checkpoint_path
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.utils.wandb import add_wandb_tags
    from mjlab.utils.wrappers import VideoRecorder

    # Get device and seed (same logic as mjlab's run_train)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
      device = "cpu"
      seed = cfg.agent.seed
      rank = 0
    else:
      local_rank = int(os.environ.get("LOCAL_RANK", "0"))
      rank = int(os.environ.get("RANK", "0"))
      os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
      device = f"cuda:{local_rank}"
      seed = cfg.agent.seed + local_rank

    configure_torch_backends()

    cfg.agent.seed = seed
    if hasattr(cfg.env, "seed"):
      cfg.env.seed = seed

    print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

    # Handle tracking tasks (check for motion file)
    is_tracking = isinstance(cfg.env, MyoSuiteTrackingEnvCfg) or (
      hasattr(cfg.env, "commands")
      and cfg.env.commands is not None
      and hasattr(cfg.env.commands, "motion")
      and cfg.env.commands.motion.motion_file is not None
    )

    # Set device in config for MyoSuite
    if isinstance(cfg.env, (MyoSuiteEnvCfg, MyoSuiteTrackingEnvCfg)):
      cfg.env.device = device
      # Store task_id in config for _patched_manager_init
      object.__setattr__(cfg.env, "task_id", task_id)

    # Use ManagerBasedRlEnv for MyoSuite tasks (via patched __init__)
    from mjlab.envs import ManagerBasedRlEnv

    # Type ignore: MyoSuite configs are patched to work with ManagerBasedRlEnv
    env = ManagerBasedRlEnv(
      cfg=cfg.env,  # type: ignore[arg-type]
      device=device,
      render_mode="rgb_array" if cfg.video else None,
    )

    # ManagerBasedRlEnv now has myosuite_env attribute set by _patched_manager_init
    # Set clip_actions on the MyoSuite environment if available
    if hasattr(env, "myosuite_env"):
      myosuite_env = env.myosuite_env
      # Find MyoSuiteVecEnvWrapper in the chain
      current = myosuite_env
      while current is not None:
        if isinstance(current, MyoSuiteVecEnvWrapper):
          current.clip_actions = cfg.agent.clip_actions
          break
        if hasattr(current, "env"):
          current = current.env
        elif hasattr(current, "unwrapped"):
          current = current.unwrapped
        else:
          break
    else:
      # Fallback: wrap with RslRlVecEnvWrapper if MyoSuite wrapper not found
      env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    # Handle resume path (same logic as mjlab)
    log_root_path = log_dir.parent
    resume_path: Path | None = None
    if cfg.agent.resume:
      if cfg.wandb_run_path is not None:
        from mjlab.utils.os import get_wandb_checkpoint_path

        resume_path, was_cached = get_wandb_checkpoint_path(
          log_root_path, Path(cfg.wandb_run_path)
        )
        if rank == 0:
          run_id = resume_path.parent.name
          checkpoint_name = resume_path.name
          cached_str = "cached" if was_cached else "downloaded"
          print(
            f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
            f"(run: {run_id}, {cached_str})"
          )
      else:
        resume_path = get_checkpoint_path(
          log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
        )

    # Handle video recording (same logic as mjlab)
    if cfg.video and rank == 0:
      # Type ignore: VideoRecorder accepts VecEnv interface, which ManagerBasedRlEnv implements
      env = VideoRecorder(
        env,  # type: ignore[arg-type]
        video_folder=Path(log_dir) / "videos" / "train",
        step_trigger=lambda step: step % cfg.video_interval == 0,
        video_length=cfg.video_length,
        disable_logger=True,
      )
      print("[INFO] Recording videos during training.")

    # Get runner class
    agent_cfg = asdict(cfg.agent)
    env_cfg = asdict(cfg.env)

    # Use MyoSuite-specific runner for tracking or regular tasks
    # Type ignore: ManagerBasedRlEnv is wrapped/compatible with VecEnv interface
    # The env might be wrapped with RslRlVecEnvWrapper or VideoRecorder, but runners accept VecEnv
    if is_tracking:
      runner = MyoSuiteMotionTrackingOnPolicyRunner(
        env,
        agent_cfg,
        str(log_dir),
        device,
        None,  # type: ignore[arg-type]
      )
    else:
      runner = MyoSuiteOnPolicyRunner(env, agent_cfg, str(log_dir), device)  # type: ignore[arg-type]

    # Add wandb tags and git repo (same logic as mjlab)
    add_wandb_tags(cfg.agent.wandb_tags)
    runner.add_git_repo_to_log(__file__)

    if resume_path is not None:
      print(f"[INFO]: Loading model checkpoint from: {resume_path}")
      runner.load(str(resume_path))

    # Write config files (same logic as mjlab)
    if rank == 0:
      dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
      dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

    # Run training (same logic as mjlab)
    runner.learn(
      num_learning_iterations=cfg.agent.max_iterations,
      init_at_random_ep_len=True,
    )

    env.close()
  else:
    # Not a MyoSuite task, use original run_train
    return _original_run_train(task_id, cfg, log_dir)


# Patch mjlab's run_train with our version
mjlab_train_module.run_train = _patched_run_train

# Now import mjlab's native train script
# Create a custom TrainConfig that extends mjlab's for backward compatibility
# This adds motion_file parameter that tests and old code expect


@dataclass(frozen=True)
class TrainConfig(MjlabTrainConfig):
  """Extended TrainConfig with motion_file for backward compatibility.

  This extends mjlab's TrainConfig to add motion_file parameter
  that was in the original custom script. For MyoSuite tasks,
  motion_file can be set directly, or use --env.commands.motion.motion-file
  (mjlab's native way).
  """

  motion_file: str | None = None
  """Motion file path for tracking tasks (backward compatibility).

    Note: mjlab's native way is to use --env.commands.motion.motion-file,
    but this parameter is kept for backward compatibility with tests and old code.
    """
  device: str = "cuda:0"
  """Device to use for training (backward compatibility).

    Note: mjlab's native way is to use --gpu-ids, but this parameter
    is kept for backward compatibility with tests and old code.
    """
  enable_nan_guard: bool = False
  """Enable NaN guard (backward compatibility).

    Note: This is already in mjlab's TrainConfig, but we keep it
    for backward compatibility.
    """

  def __post_init__(self):
    """Initialize motion_file from env.commands if available."""
    # Call parent __post_init__ if it exists
    if hasattr(super(), "__post_init__"):
      super().__post_init__()

    # If motion_file is not set but env.commands.motion.motion_file is,
    # extract it for backward compatibility
    if (
      self.motion_file is None
      and hasattr(self.env, "commands")
      and self.env.commands is not None
      and hasattr(self.env.commands, "motion")
      and hasattr(self.env.commands.motion, "motion_file")
      and self.env.commands.motion.motion_file is not None
    ):
      # Use object.__setattr__ for frozen dataclass
      object.__setattr__(self, "motion_file", self.env.commands.motion.motion_file)


# Re-export run_train for backward compatibility with tests and other code
# Note: run_train is our patched version that handles MyoSuite tasks
run_train = _patched_run_train


def wrapper_main():
  """Wrapper main that ensures registration before calling mjlab's main."""
  mjlab_main()


# Alias main to wrapper_main for backward compatibility
main = wrapper_main
__all__ = ["main", "TrainConfig", "run_train", "wrapper_main"]


if __name__ == "__main__":
  wrapper_main()
