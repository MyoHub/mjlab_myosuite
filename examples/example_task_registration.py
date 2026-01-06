"""Example: Registering a specific MyoSuite task with mjlab.

This example demonstrates how to register a MyoSuite environment following
mjlab's native task registration pattern, as shown in the tutorial:
https://github.com/mujocolab/mjlab/blob/main/notebooks/create_new_task.ipynb

Usage:
    python examples/example_task_registration.py
"""

# Example 1: Using mjlab's native registration (recommended)
try:
  from mjlab.tasks.registry import register_mjlab_task
  from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

  from mjlab_myosuite.config import MyoSuiteEnvCfg, get_default_myosuite_rl_cfg

  def myoelbow_env_cfg(play: bool = False):
    """Environment configuration for MyoElbow task."""
    cfg = MyoSuiteEnvCfg()
    cfg.num_envs = 4096 if not play else 1  # More envs for training
    cfg.device = "cuda:0" if not play else "cpu"  # GPU for training
    return cfg

  def myoelbow_rl_cfg():
    """RL configuration for MyoElbow task."""
    cfg = get_default_myosuite_rl_cfg()
    cfg.experiment_name = "myoelbow"
    cfg.max_iterations = 2000
    return cfg

  # Register the task
  register_mjlab_task(
    task_id="Mjlab-MyoElbow-v0",
    env_cfg=myoelbow_env_cfg(play=False),  # type: ignore[arg-type]
    play_env_cfg=myoelbow_env_cfg(play=True),  # type: ignore[arg-type]
    rl_cfg=myoelbow_rl_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )

  print("✅ Registered MyoElbow task with mjlab's native system")

except ImportError as e:
  print(f"⚠️  mjlab native registration not available: {e}")
  print("   Falling back to gymnasium registry...")

  # Example 2: Using gymnasium registry (fallback)
  import gymnasium as gym

  gym.register(
    id="Mjlab-MyoElbow-v0",
    entry_point="mjlab_myosuite.env_factory:make_myosuite_env",
    disable_env_checker=True,
    kwargs={
      "myosuite_env_id": "myoElbowPose1D6MRandom-v0",
      "env_cfg_entry_point": "mjlab_myosuite.config:MyoSuiteEnvCfg",
      "rl_cfg_entry_point": "mjlab_myosuite.config:get_default_myosuite_rl_cfg",
    },
  )

  print("✅ Registered MyoElbow task with gymnasium registry")

# Example 3: Using the auto-registration (simplest)
# Just import mjlab_myosuite and all MyoSuite environments are registered automatically
import mjlab_myosuite  # noqa: F401

print("\n📝 Usage:")
print("  # Train:")
print(
  "  uv run python scripts/train.py Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 --agent.max-iterations 2000"
)
print("\n  # Play:")
print(
  "  uv run python scripts/play.py Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 --checkpoint_file <path>"
)
