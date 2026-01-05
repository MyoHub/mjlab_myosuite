# 🚧 Early Prototype — Community Feedback Welcome!

This project is currently in an early prototype stage.

Features, architecture, and documentation are actively evolving, and breaking changes are likely as we iterate.

We’re building this openly with the community, so feedback, ideas, and contributions are highly encouraged! If you’d like to help shape the direction of the project:

Open an issue to share suggestions or report bugs
Start a discussion about improvements
Submit a pull request with enhancements Thank you for helping us improve this project!

# MyoSuite to mjlab Integration

Integration package for using MyoSuite environments with mjlab's training infrastructure.

## Features

- ✅ **Automatic Registration**: All MyoSuite environments are automatically registered with mjlab
- ✅ **MJX/Warp Support**: Compatible with both standard MyoSuite and mjx/warp GPU-accelerated versions
- ✅ **Native mjlab Integration**: Uses mjlab's native task registration when available
- ✅ **Backward Compatible**: Falls back to gymnasium registry if mjlab native registration unavailable
- ✅ **Full Test Coverage**: Comprehensive unit tests for all functionality

## Installation

```bash
# Install mjlab-myosuite
pip install -e .

# Or with uv (faster)
uv pip install -e .
```

## Quick Start

### 1. Basic Usage (gym based)

```python
import gymnasium as gym
import mjlab_myosuite  # Auto-registers all MyoSuite environments

# Create a MyoSuite environment wrapped for mjlab
env = gym.make("Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0")
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
env.close()
```

### 2. Training with mjlab

**Use the local training scripts** (includes workarounds for mjlab compatibility issues):

```bash
# Train a policy
uv run python scripts/train.py Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 \
    --agent.max-iterations 2000 \
    --agent.num-steps-per-env 24

# Play with trained policy
uv run python scripts/play.py Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 \
    --checkpoint_file logs/rsl_rl/myosuite/.../model_2000.pt
```

**Note**: The local scripts (`scripts/train.py` and `scripts/play.py`) automatically handle:

- ✅ Fallback for missing mjlab utilities
- ✅ Proper MyoSuite environment registration timing
- ✅ Compatibility with different mjlab versions

### 3. Custom Task Registration

For registering specific tasks with custom configurations, see:

- `examples/example_task_registration.py` - Complete example following mjlab's tutorial pattern

## Architecture

The integration follows mjlab's native task registration pattern from the [create_new_task tutorial](https://github.com/mujocolab/mjlab/blob/main/notebooks/create_new_task.ipynb):

```
┌─────────────────────────────────────┐
│   mjlab Training Pipeline           │
│   (PPO, WandB, etc.)                │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   MyoSuite Wrapper                  │
│   - Adapts Gym API to mjlab         │
│   - Handles batched operations      │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   MyoSuite (Standard or MJX/Warp)   │
│   - Musculoskeletal models          │
│   - Task-specific rewards          │
└─────────────────────────────────────┘
```

## Supported MyoSuite Versions

- **Standard MyoSuite**: CPU-based MuJoCo simulation
- **MJX/Warp MyoSuite**: GPU-accelerated from the [mjx branch](https://github.com/MyoHub/myosuite/tree/mjx/myosuite)

The wrapper automatically detects and supports both versions.

## Configuration

### Environment Configuration

```python
from mjlab_myosuite.config import MyoSuiteEnvCfg

cfg = MyoSuiteEnvCfg()
cfg.num_envs = 4096  # For training
cfg.device = "cuda:0"  # Use GPU for mjx/warp versions
```

### RL Configuration

```python
from mjlab_myosuite.config import get_default_myosuite_rl_cfg

rl_cfg = get_default_myosuite_rl_cfg()
rl_cfg.max_iterations = 2000
rl_cfg.algorithm.learning_rate = 3e-4
```

## Known Issues

### mujoco_warp Import Error

If you see an error like:

```
ImportError: cannot import name 'rays' from 'mujoco_warp'
```

This is a known compatibility issue between mjlab and mujoco_warp. The mjlab codebase tries to import `rays` but mujoco_warp exports `ray` instead.

**Solution**: The local training scripts (`scripts/train.py` and `scripts/play.py`) automatically apply a monkey patch to fix this issue. Always use these scripts instead of mjlab's native scripts:

```bash
# ✅ Use local scripts (includes workarounds)
python scripts/train.py Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 ...

# ❌ Don't use mjlab's native script (will fail with import error)
python -m mjlab.scripts.train Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 ...
```

### Environment Registration Timing

If you see "No tasks found with prefix 'Mjlab-'", ensure that:

1. MyoSuite is properly installed
2. You're using the local scripts (`scripts/train.py`), not mjlab's native scripts
3. The import of `mjlab_myosuite` happens before any mjlab imports

The local scripts handle this automatically.

## ONNX Model Export

MyoSuite environments support ONNX model export for deployment and inference. The `MyoSuiteOnPolicyRunner` automatically exports ONNX models when using wandb logging:

```python
from mjlab_myosuite.rl.runner import MyoSuiteOnPolicyRunner

# During training, ONNX models are automatically exported
runner = MyoSuiteOnPolicyRunner(env, agent_cfg, log_dir, device)
runner.learn(num_learning_iterations=1000)
# ONNX model is saved alongside the PyTorch checkpoint
```

The exported ONNX model includes:

- Policy network (actor) with optional observation normalizer
- MyoSuite-specific metadata (action dimensions, observation dimensions, etc.)
- Compatibility with ManagerBasedRlEnv structure

## Viser Playback Utility

The `playback_with_viser` utility provides a convenient way to visualize policy execution using the Viser web-based viewer:

```python
from scripts.play import playback_with_viser
import gymnasium as gym

# Create environment and policy
env = gym.make("Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0")
policy = load_policy("path/to/checkpoint.pt")

# Playback with Viser
playback_with_viser(env, policy, verbose=True)
```

You can also use it from the command line:

```bash
# Use Viser viewer explicitly
python scripts/play.py Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 \
    --viewer viser \
    --checkpoint_file logs/rsl_rl/myosuite/.../model_2000.pt

# Specify Viser server port
python scripts/play.py Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 \
    --viewer viser \
    --viser-port 8080 \
    --checkpoint_file logs/rsl_rl/myosuite/.../model_2000.pt
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_myosuite_integration.py::test_wrapper_creation_direct

# Run GPU acceleration tests (requires CUDA)
pytest tests/test_gpu_acceleration.py -v

# Run ONNX export tests (requires ONNX)
pytest tests/test_onnx_export.py -v
```

## Documentation

- [mjlab Tutorial](https://github.com/mujocolab/mjlab/blob/main/notebooks/create_new_task.ipynb) - Official mjlab task creation tutorial
- [MyoSuite Documentation](https://myosuite.readthedocs.io/) - MyoSuite documentation
