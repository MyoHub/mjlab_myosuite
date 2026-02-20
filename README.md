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
- ✅ **Factory API**: Create envs with `make_myosuite_env()` (no gym registration required)
- ✅ **Full Test Coverage**: Comprehensive unit tests for all functionality

## Installation

```bash
# Install mjlab-myosuite
pip install -e .
pip install "myosuite @ git+https://github.com/MyoHub/myosuite.git@mjx"

# Or with uv (faster)
uv venv
uv pip install -e .
uv pip install "myosuite @ git+https://github.com/MyoHub/myosuite.git@mjx"
```

## Quick Start

### 1. Basic Usage

```python
from mjlab_myosuite.env_factory import make_myosuite_env

# Create a MyoSuite environment wrapped for mjlab
env = make_myosuite_env("myoElbowPose1D6MRandom-v0")
obs, info = env.reset()
action = env.action_space.sample()
obs, rewards, dones, extras = env.step(action)
env.close()
```

### 2. Training with mjlab

**Use the mjlab training scripts**:

```bash
# Train a policy
uv run train Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 \
    --agent.max-iterations 200 \
    --agent.num-steps-per-env 512

# Play with trained policy
uv run play Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 \
    --checkpoint_file logs/rsl_rl/myosuite/.../model_199.pt
```

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
│   - Task-specific rewards           │
└─────────────────────────────────────┘
```

## Supported MyoSuite Versions

- **Standard MyoSuite**: CPU-based MuJoCo simulation (default).
- **MJX/Warp MyoSuite**: GPU-accelerated from the [MyoSuite mjx branch](https://github.com/MyoHub/myosuite/tree/mjx). Install with: `git clone https://github.com/MyoHub/myosuite.git && cd myosuite && git checkout mjx && pip install -e .`

Set `cfg.physics_backend = PhysicsBackend.WARP` (or use auto-detection) when using the mjx branch so the data path can align with mjlab's Warp bridge when available.

The wrapper automatically detects and supports both versions.

## Configuration

### Environment Configuration

```python
from mjlab_myosuite.config import MyoSuiteEnvCfg, PhysicsBackend

cfg = MyoSuiteEnvCfg()
cfg.num_envs = 4096  # For training
cfg.device = "cuda:0"  # Use GPU for mjx/warp versions
cfg.physics_backend = PhysicsBackend.WARP  # Optional: use when MyoSuite is from mjx branch
```

### RL Configuration

```python
from mjlab_myosuite.config import get_default_myosuite_rl_cfg

rl_cfg = get_default_myosuite_rl_cfg()
rl_cfg.max_iterations = 2000
rl_cfg.algorithm.learning_rate = 3e-4
```

## Known Issues

TBD

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

## Tracking Tasks (not yet fully implemented)

MyoSuite tracking tasks follow the same structure as mjlab's tracking tasks, allowing you to train policies to track reference motions. The tracking functionality is implemented in `src/mjlab_myosuite/tasks/tracking/` following the [mjlab tracking structure](https://github.com/mujocolab/mjlab/tree/main/src/mjlab/tasks/tracking).

### Tracking Configuration

```python
from mjlab_myosuite.tasks.tracking.tracking_env_cfg import MyoSuiteTrackingEnvCfg

# Create tracking configuration
cfg = MyoSuiteTrackingEnvCfg()
cfg.num_envs = 4096
cfg.device = "cuda:0"
cfg.commands.motion.motion_file = "path/to/motion.npz"
```

### Training Tracking Tasks

```bash
# Train with motion file from wandb artifact
uv run train Mjlab-MyoSuite-Tracking-myoElbowPose1D6MRandom-v0 \
    --motion-file examples/elbow_sinusoidal_motion.npz     \
    --agent.max-iterations 10000
```

### Playing Tracking Tasks

```bash
# Play with motion file
uv run play Mjlab-MyoSuite-Tracking-myoElbowPose1D6MRandom-v0 \
    --checkpoint_file logs/rsl_rl/myosuite/.../model_2000.pt \
    --motion-file path/to/motion.npz
```

The tracking runner (`MyoSuiteMotionTrackingOnPolicyRunner`) extends the base MyoSuite runner and provides support for motion tracking, including wandb artifact integration for motion files.

## Viser Playback Utility

The `playback_with_viser` utility provides a convenient way to visualize policy execution using the Viser web-based viewer:

```python
from mjlab_myosuite.env_factory import make_myosuite_env
from scripts.play import playback_with_viser

# Create environment and policy
env = make_myosuite_env("myoElbowPose1D6MRandom-v0")
policy = load_policy("path/to/checkpoint.pt")

# Playback with Viser
playback_with_viser(env, policy, verbose=True)
```

You can also use it from the command line:

```bash
# Use Viser viewer explicitly
play Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 \
    --viewer viser \
    --checkpoint_file logs/rsl_rl/myosuite/.../model_2000.pt

# Specify Viser server port
play Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 \
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

# Verify mjlab_myosuite operates like mjlab (reset/step/get_observations, device, scaling)
pytest tests/test_mjlab_env_contract.py tests/test_gpu_scaling.py -v

# Run ONNX export tests (requires ONNX)
pytest tests/test_onnx_export.py -v
```

## Development

**Benchmarks** (verify GPU use and scaling with num_envs):

```bash
# Throughput at different num_envs (CPU or GPU)
uv run python benchmarks/run_benchmarks.py --device cuda:0 --num-envs 64 256 1024

# Compare mjlab_myosuite vs native mjlab on GPU (both should scale similarly)
uv run python benchmarks/run_benchmarks.py --compare-mjlab --num-envs 64 256 1024

# Assert that throughput scales with num_envs
uv run python benchmarks/run_benchmarks.py --verify-scaling --num-envs 64 256
```

Run tests:

```bash
make test          # Run all tests
make test-fast     # Skip slow integration tests
uv run --no-default-groups --group cu128 --group dev pyright
uv run --no-default-groups --group cu128 --group dev pytest
```

Format code:

```bash
uvx pre-commit install
make format
```

## Documentation

- [mjlab Tutorial](https://github.com/mujocolab/mjlab/blob/main/notebooks/create_new_task.ipynb) - Official mjlab task creation tutorial
- [MyoSuite Documentation](https://myosuite.readthedocs.io/) - MyoSuite documentation
