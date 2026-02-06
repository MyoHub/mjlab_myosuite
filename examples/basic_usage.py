"""Basic usage: create env, run one episode, inspect observations and actions.

Run: uv run python examples/basic_usage.py
Requires: MyoSuite installed (e.g. pip install myosuite @ git+https://github.com/MyoHub/myosuite.git@mjx)
"""

from mjlab_myosuite.env_factory import make_myosuite_env

# Create wrapped MyoSuite environment (single env by default)
env = make_myosuite_env("myoElbowPose1D6MRandom-v0")
obs, info = env.reset(seed=42)

# Inspect spaces
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# Single episode rollout
done, step = False, 0
while not done and step < 500:
  action = env.action_space.sample()
  obs, rewards, dones, extras = env.step(action)
  done = bool(dones.any()) if hasattr(dones, "any") else dones
  step += 1
print(f"Episode finished in {step} steps, last reward: {rewards}")

env.close()
