# Analysis: GPU Computing in MuJoCo-Based RL Environments

## Three Codebases Compared

| | **myosuite `mjx` branch** | **mjlab** | **mjlab_myosuite `refactor_code`** |
|---|---|---|---|
| **Source** | [MyoHub/myosuite@mjx](https://github.com/MyoHub/myosuite/tree/mjx) | [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | [MyoHub/mjlab_myosuite@refactor_code](https://github.com/MyoHub/mjlab_myosuite/tree/refactor_code) |
| **Paradigm** | Native GPU physics (MJX + JAX) | CPU physics + GPU tensor bridge (MuJoCo Warp + PyTorch) | CPU physics + GPU tensor shuttle (MuJoCo + PyTorch wrapper) |
| **Physics engine** | MJX (MuJoCo compiled to XLA/Warp kernels) | MuJoCo C (CPU) bridged to GPU via `mujoco_warp` | Standard MuJoCo C (CPU only) |
| **Array framework** | JAX (`jax.numpy`) | PyTorch via zero-copy `WarpBridge`/`TorchArray` | NumPy → PyTorch (`torch.as_tensor`) |
| **Vectorization** | `jax.vmap` over single GPU model | N replicated `mj_model`/`mj_data`, tiled via GPU kernel | `gymnasium.vector.SyncVectorEnv` (N CPU copies) |
| **Kernel optimization** | `jax.jit` (XLA compilation) | CUDA graph capture (`wp.ScopedCapture`) | None |
| **RL framework** | Brax PPO (JAX-native) | RSL-RL PPO (PyTorch) | RSL-RL PPO (PyTorch) |
| **Typical num_envs** | 4096 (single GPU kernel) | 16–4096 (CPU-limited physics, GPU tensors) | 1–8 (fully CPU-bound) |

---

## 1. myosuite `mjx` — Physics on GPU

### Core Mechanism

The entire MuJoCo physics solver is compiled to GPU kernels:

```python
# playground_pose_v0.py
self._mjx_model = mjx.put_model(self._mj_model, impl="warp")
```

This converts the MuJoCo model into NVIDIA Warp or XLA GPU kernels. From this point **all computation stays on GPU**.

### GPU Computing Characteristics

- **Pure functional** — `reset()` and `step()` are side-effect-free JAX functions operating on `jax.numpy` arrays in GPU memory.
- **Batch vectorization** — `jax.vmap` maps the scalar step function across 4096 environments as a single GPU kernel launch. No Python loop.
- **JIT compilation** — `jax.jit` traces the entire reset→step→reward→observation pipeline and compiles it to an optimized XLA/CUDA program. First call is slow (seconds); subsequent calls execute the compiled kernel.
- **Zero CPU↔GPU sync** during training — data never leaves GPU between env step and policy gradient update.
- **Key-based PRNG** — `jax.random.split(rng, 3)` provides deterministic, stateless randomness compatible with GPU parallelism.
- **Autodifferentiation** — JAX's autograd works through the physics, enabling gradient-based optimization and differentiable simulation.

### Physics Optimizations

```python
self._mj_model.geom_margin = np.zeros(...)    # reduce collision overhead
self._mj_model.opt.iterations = 6             # fewer solver iterations
self._mj_model.opt.ls_iterations = 6
# Disable expensive cylinder collisions on GPU:
for geom in spec.geoms:
    if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        geom.conaffinity = 0; geom.contype = 0
# Pre-allocate contact buffers for batch:
data = make_data(..., nconmax=125 * num_envs, impl="warp")
```

### Data Flow

```
GPU: mjx.step() → jp.array obs/reward → Brax PPO update → repeat
     ↑ never leaves GPU ↑
```

### Reward / Observation Computation

All vectorized JAX on GPU:
```python
pose_err = state.info['target_angles'] - data.qpos
pose_dist = jp.linalg.norm(pose_err, axis=-1)
reward = -pose_dist * weight + jp.where(pose_dist < threshold, bonus, 0.)
obs = jp.concatenate([data.qpos, data.qvel * dt, data.act, pose_err])
```

### Environment Registry

`MjxElbowPoseFixed-v0`, `MjxElbowPoseRandom-v0`, `MjxFingerPoseFixed-v0`, `MjxFingerPoseRandom-v0`, `MjxHandReachFixed-v0`, `MjxHandReachRandom-v0` — all configured with `num_envs=4096`, `ctrl_dt=0.02`, `sim_dt=0.002` (10 substeps).

### Training

```python
from brax.training.agents.ppo import train as ppo
make_inference_fn, params, _ = ppo.train(
    environment=env, num_envs=4096,
    wrap_env_fn=wrapper.wrap_for_brax_training, **ppo_params)
```

PPO: 40M timesteps, batch_size=512, 32 minibatches, 3×(64) networks.

---

## 2. mjlab — CPU Physics + GPU Tensor Bridge with CUDA Graphs

### Core Mechanism

MuJoCo's C physics solver runs on CPU. Results are bridged to GPU via `mujoco_warp` (mjwarp) with **zero-copy** Warp↔PyTorch sharing and optional **CUDA graph capture**.

```python
# sim.py — CUDA graph capture
with wp.ScopedCapture() as capture:
    mjwarp.step(self.wp_model, self.wp_data)
self.step_graph = capture.graph

# Replay captured graph (single kernel launch replays entire sequence)
wp.capture_launch(self.step_graph)
```

### GPU Computing Characteristics

- **Zero-copy data bridge** — `WarpBridge` wraps Warp GPU arrays; `TorchArray` exposes them as PyTorch tensors sharing the same GPU memory. No data copying between frameworks:
  ```python
  # sim_data.py
  class TorchArray:
      def __init__(self, wp_array):
          self._tensor = wp.to_torch(wp_array)  # shared GPU memory
  class WarpBridge:
      def __getattr__(self, name):
          return TorchArray(getattr(self._struct, name))  # zero-copy
      def __setattr__(self, name, val):
          raise AttributeError(...)  # read-only: preserves memory addresses for CUDA graphs
  ```

- **CUDA graph capture** — Records the GPU kernel sequence for `step()`, `forward()`, and `reset()` operations. Replayed as a single kernel launch, eliminating per-step launch overhead. Requires CUDA driver ≥12.4 with memory pool support. Falls back to direct kernel calls on CPU.

- **Model field expansion** — For domain randomization, `expand_model_fields()` tiles MuJoCo model arrays across N worlds via a GPU kernel (modulo arithmetic). Triggers immediate graph re-capture since pointers change.

- **Batched entity tensors** — `EntityData` maintains PyTorch tensors in shape `(nworld, feature_dim)` for root state (13D), joint state, control targets, external wrenches, etc. All RL manager computations operate on these batched tensors.

- **Manager-based architecture** — Composable managers for observations, rewards, actions, commands, events, terminations, and curriculum. Each manager processes batched GPU tensors:
  - **ObservationManager**: compute → noise → clip → scale → delay → history (all on GPU tensors)
  - **RewardManager**: weighted term aggregation with optional `scale_by_dt`
  - **ActionManager**: action distribution → history update → apply to simulation
  - **EventManager**: domain randomization (startup / reset / interval triggers)
  - **TerminationManager**: per-environment done flags (terminated vs truncated)
  - **CurriculumManager**: progressive difficulty modification

- **GPU-accelerated sensors** — Raycast sensor uses BVH-accelerated Warp kernels for terrain sensing. PD actuators run parallel control loops via PyTorch tensor ops.

- **NaN guarding** — Numerical stability detection with configurable modes (disabled/warn/sanitize/error).

- **TF32 precision** — Optional reduced-precision mode (10-bit mantissa vs 23-bit FP32) for speed.

### Data Flow

```
CPU: MuJoCo C solver (mj_step)
       ↓ mujoco_warp bridge
GPU: Warp arrays ←zero-copy→ PyTorch tensors (WarpBridge/TorchArray)
       ↓
GPU: CUDA graph replay (step/forward/reset)
       ↓
GPU: Manager computations (obs/reward/action — all PyTorch tensor ops)
       ↓
GPU: RSL-RL policy update
       ↓
GPU: Action tensors → apply to simulation
```

### Step Execution (`ManagerBasedRlEnv.step()`)

```python
def step(self, action):
    self.action_manager.process_action(action)
    for _ in range(self.cfg.decimation):
        self.scene.write_data_to_sim()
        self.sim.step()                      # CUDA graph or direct mjwarp call
        self.scene.update(dt)
    self.sim.forward()                       # single forward kinematics pass
    self.termination_manager.compute()       # batched GPU tensor ops
    self.reward_manager.compute(dt)          # batched GPU tensor ops
    self._reset_idx(terminated_envs)
    self.observation_manager.compute()       # compute → noise → clip → scale → delay → history
    return obs, reward, terminated, truncated, info
```

### Environment Examples

Velocity tracking task: observations include lin/ang velocity, joint states, terrain heights; rewards for velocity tracking, uprightness, penalties for joint limits/action rate; domain randomization of friction, encoder bias, COM shifts; curriculum on terrain difficulty.

### Training

```python
# RSL-RL OnPolicyRunner via RslRlVecEnvWrapper
RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(actor_hidden_dims=(256,256), critic_hidden_dims=(256,256)),
    algorithm=RslRlPpoAlgorithmCfg(learning_rate=3e-4, num_learning_epochs=5, num_mini_batches=4),
)
```

---

## 3. mjlab_myosuite — CPU Physics + GPU Tensor Shuttle (Wrapper)

### Core Mechanism

Standard MuJoCo C library runs physics on CPU. The wrapper converts NumPy outputs to PyTorch tensors on the target GPU device.

```python
# vec_env_wrapper.py
class MyoSuiteVecEnvWrapper(VecEnv, gym.Env):
    step_result = self.env.step(actions_np)              # CPU physics
    rew_tensor = torch.as_tensor(rew, device=self.device)  # → GPU
```

### GPU Computing Characteristics

- **`SyncVectorEnv`** — Creates N deep-copied MyoSuite environments, steps them **sequentially on CPU**.
- **Pinned memory transfers** — Pre-allocated `pin_memory=True` buffers for faster CPU→GPU DMA:
  ```python
  self._pinned_policy_buffer = torch.empty(shape, pin_memory=True)
  self._pinned_policy_buffer.copy_(torch.from_numpy(arr))
  return self._pinned_policy_buffer.to(device=self.device, non_blocking=True)
  ```
- **`non_blocking=True`** — Asynchronous GPU transfers overlap with CPU work.
- **No JIT, no CUDA graphs, no zero-copy** — Pure CPU→GPU data copying each step.
- **MockSim compatibility layer** — `sim_compat.py` creates mock `Simulation` objects that pass `isinstance` checks, exposing `mj_model`, `mj_data`, `wp_data` for mjlab's viewers (Viser, offscreen). `MockWpData` wraps NumPy arrays with `.numpy()` method and batch dimension.
- **DataAdapter** — Wraps `mj_data` as PyTorch tensors with batch dimension for `OffscreenRenderer`:
  ```python
  class DataAdapter:
      @property
      def qpos(self):
          return torch.from_numpy(self._mj_data.qpos.copy()).unsqueeze(0)
  ```

### Data Flow

```
CPU: mujoco.step() → numpy obs/reward
       ↓ pinned memory transfer (async)
GPU: torch tensors → RSL-RL policy update → torch actions
       ↓ .cpu().numpy() sync
CPU: actions_np → mujoco.step() ...
```

### Compatibility Infrastructure

Extensive mocking to bridge MyoSuite into mjlab's ecosystem:
- `MockSimPrimary` — Inherits `mjlab.sim.sim.Simulation` for isinstance checks
- `MockWpData` — Batched NumPy→ArrayProxy for Viser viewer
- `_MockActionManager`, `_MockObservationManager`, `_MockScene` — For ONNX export
- Supports both standard MyoSuite (`mj_model`) and MJX versions (`model`) attribute names

### Training

```python
RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(actor_hidden_dims=(256,256), critic_hidden_dims=(256,256)),
    algorithm=RslRlPpoAlgorithmCfg(learning_rate=3e-4, num_learning_epochs=5, num_mini_batches=4),
)
```

---

## Full Comparison Table

| Dimension | **myosuite `mjx`** | **mjlab** | **mjlab_myosuite** |
|---|---|---|---|
| **Physics location** | GPU (MJX compiled kernels) | CPU (MuJoCo C) + GPU bridge (mjwarp) | CPU (MuJoCo C) only |
| **GPU data access** | Native JAX arrays on GPU | Zero-copy Warp↔PyTorch (`TorchArray`) | Copy: NumPy → pinned → GPU tensor |
| **Kernel optimization** | `jax.jit` (XLA compilation) | CUDA graph capture (`wp.ScopedCapture`) | None |
| **Vectorization** | `jax.vmap` — single model, one kernel | N tiled models via `expand_model_fields()` | `SyncVectorEnv` — N sequential CPU copies |
| **Scalable batch size** | 4096+ on one GPU | 16–4096 (CPU physics is bottleneck) | 1–64 (fully CPU-bound) |
| **CPU↔GPU transfers** | None during training | Minimal (zero-copy bridge, CUDA graphs) | Every step: obs→GPU, actions→CPU |
| **State management** | Functional (`state.replace()`) | In-place tensor mutation (CUDA graph safe) | Object-oriented (`self.env.step()`) |
| **Autodiff through physics** | Yes (JAX autograd) | No | No |
| **JIT compilation** | Yes (`jax.jit`) | No (CUDA graphs serve similar purpose) | No |
| **Domain randomization** | Manual per-env (`jp.where`) | `EventManager` + `expand_model_fields()` | Not built-in |
| **Observation pipeline** | Manual `jp.concatenate` | Manager: compute→noise→clip→scale→delay→history | Manual torch concatenation |
| **Reward pipeline** | Manual `jp.where` + weights | Manager: weighted terms, dt-scaling, NaN guard | Manual torch computation |
| **Action pipeline** | Sigmoid normalization | Manager: process→history→distribute→apply | Pass-through |
| **Sensor support** | Site positions only | Raycast (BVH/Warp), contact, cameras | MuJoCo sensor API |
| **PRNG** | Key-based (`jax.random.split`) | PyTorch (`torch.manual_seed`) | Seed-based (`np.random`) |
| **Reset handling** | Vectorized `jp.where` conditional | Manager-orchestrated, per-env flags | `SyncVectorEnv` auto-reset |
| **NaN handling** | None explicit | Configurable: disabled/warn/sanitize/error | None |
| **ONNX export** | Not integrated | Native via `MjlabOnPolicyRunner` | Built-in via mjlab mock managers |
| **Viewer** | Custom visualization scripts | Viser (web), offscreen, native MuJoCo | Full mjlab stack via mock sim |
| **Curriculum** | Not present | `CurriculumManager` | Not present |
| **Terrain** | Not present | Terrain generation + raycast sensing | Not present |
| **Motion tracking** | Not present | Via command manager | Full tracking task support |
| **Dependencies** | JAX, Brax, Flax, Optax | PyTorch, mujoco_warp, RSL-RL, Warp | PyTorch, RSL-RL, mjlab, MyoSuite |

---

## Similarities Across All Three

1. **MuJoCo as physics foundation** — All three use MuJoCo XML models, just at different compilation targets.
2. **PPO as default RL algorithm** — Brax PPO (JAX) / RSL-RL PPO (PyTorch) with similar hyperparameters (LR=3e-4, 2-3 hidden layers).
3. **Observation structure** — Joint positions, velocities, actuator activations, task-specific error signals.
4. **Multi-component rewards** — Tracking error + regularization + bonus/penalty thresholds.
5. **Batched environment design** — All support multiple parallel environments (differ in mechanism and scale).
6. **Headless rendering** — `MUJOCO_GL=egl` for server environments.
7. **Warp awareness** — MJX uses `impl="warp"` for kernels; mjlab uses native `mujoco_warp`; mjlab_myosuite provides mock `wp_data`.

## Key Architectural Differences

### GPU Execution Model

**MJX (JAX/XLA)**: Traces the entire Python function → compiles to a single fused GPU program → executes repeatedly. The GPU program includes physics, reward, observation, and reset logic. No Python overhead after compilation.

**mjlab (Warp/CUDA Graphs)**: Records sequences of Warp GPU kernel launches → replays as a single CUDA graph. Physics kernels run individually but launch overhead is amortized. Manager computations use standard PyTorch tensor ops (not captured in graphs).

**mjlab_myosuite (PyTorch copy)**: No GPU kernel compilation or capture. Each step involves Python dispatch → CPU physics → memory copy → PyTorch ops → memory copy back.

### Composability vs Throughput

**MJX** maximizes throughput but reward/observation/reset logic is hand-coded per environment. Adding a new observation requires modifying the environment class.

**mjlab** sacrifices some throughput for composability. The manager architecture lets you add observation terms, reward terms, domain randomization events, and termination conditions declaratively via configuration. A new sensor observation is a config change, not a code change.

**mjlab_myosuite** inherits MyoSuite's existing environment logic and wraps it for mjlab compatibility. Minimal code changes needed to bring existing MyoSuite envs into mjlab's training pipeline.

### Memory Model

**MJX**: Immutable JAX arrays. `state.replace(data=new_data)` creates new state. Garbage collected. No aliasing concerns.

**mjlab**: In-place mutation on fixed-address GPU arrays. Required by CUDA graph capture (graphs hold pointers to specific memory locations). `WarpBridge.__setattr__` raises errors to prevent accidental pointer invalidation.

**mjlab_myosuite**: Standard NumPy/PyTorch semantics. No fixed-address constraints.

---

## Conclusion

The three projects occupy different points on a throughput↔composability spectrum:

```
Throughput                                              Composability
    |                                                        |
    MJX (4096 envs, all-GPU,         mjlab (manager-based,
    hand-coded envs)                  CUDA graphs, configurable)
                                                    |
                                          mjlab_myosuite (wrapper,
                                          bridges existing MyoSuite
                                          envs into mjlab)
```

- **MJX** is optimal when you need raw GPU simulation throughput for simple tasks with fixed observation/reward structure.
- **mjlab** is optimal for building complex RL tasks with rich sensor suites, domain randomization, curriculum learning, and terrain — where the manager architecture pays for itself in development speed.
- **mjlab_myosuite** is optimal for leveraging existing MyoSuite biomechanical environments within mjlab's training/visualization/export ecosystem without rewriting them.

The three are designed to be complementary. mjlab_myosuite's `env_factory.py` already tries MJX import paths first, and its `sim_compat.py` handles both attribute naming conventions (`mj_model` vs `model`), anticipating a future where MJX-backed environments can be wrapped in mjlab's manager infrastructure.
