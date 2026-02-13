"""Example: Registering a specific MyoSuite task with mjlab.

This example demonstrates how to register a MyoSuite environment following
mjlab's native task registration pattern, as shown in the tutorial:
https://github.com/mujocolab/mjlab/blob/main/notebooks/create_new_task.ipynb

Environments are created with make_myosuite_env() or make_myosuite_env_from_task_id();
there is no gym registration in mjlab_myosuite.

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
    cfg.num_envs = 4096 if not play else 1
    cfg.device = "cuda:0" if not play else "cpu"
    return cfg

  def myoelbow_rl_cfg():
    """RL configuration for MyoElbow task."""
    cfg = get_default_myosuite_rl_cfg()
    cfg.experiment_name = "myoelbow"
    cfg.max_iterations = 2000
    return cfg

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

# Example 2: Creating envs with make_myosuite_env (no gym registration)
from mjlab_myosuite.env_factory import make_myosuite_env

env = make_myosuite_env("myoElbowPose1D6MRandom-v0")
env.close()
print("✅ Created env with make_myosuite_env('myoElbowPose1D6MRandom-v0')")

print("\n📝 Usage:")
print("  # Train (task_id from mjlab registry):")
print(
  "  uv run python -m mjlab_myosuite.scripts.train Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 --agent.max-iterations 2000"
)
print("\n  # Play:")
print(
  "  uv run python -m mjlab_myosuite.scripts.play Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 --checkpoint_file <path>"
)
