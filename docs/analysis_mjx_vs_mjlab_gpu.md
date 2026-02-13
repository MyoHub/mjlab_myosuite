# Analysis: GPU Computing in MuJoCo-Based RL Environments

## Three Codebases Compared

| | **myosuite `mjx` branch** | **mjlab** | **mjlab_myosuite `refactor_code`** |
|---|---|---|---|
| **Source** | [MyoHub/myosuite@mjx](https://github.com/MyoHub/myosuite/tree/mjx) | [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | [MyoHub/mjlab_myosuite@refactor_code](https://github.com/MyoHub/mjlab_myosuite/tree/refactor_code) |
| **Paradigm** | Native GPU physics (MJX + JAX) | CPU physics bridged to GPU (MuJoCo Warp + CUDA Graphs + PyTorch) | CPU physics with GPU tensor shuttle (MuJoCo + PyTorch wrapper) |
| **Physics engine** | MJX (MuJoCo compiled to XLA/Warp GPU kernels) | MuJoCo C (CPU) + `mujoco_warp` GPU bridge | Standard MuJoCo C (CPU only) |
| **Array framework** | JAX (`jax.numpy`) | PyTorch via zero-copy `WarpBridge`/`TorchArray` | NumPy -> PyTorch (`torch.as_tensor`, pinned memory) |
| **Vectorization** | `jax.vmap` over single GPU model | N replicated `mj_model`/`mj_data` via `mjwarp.put_data(nworld=N)` | `gymnasium.vector.SyncVectorEnv` (N deep-copied CPU envs) |
| **Kernel optimization** | `jax.jit` (full XLA trace compilation) | CUDA graph capture (`wp.ScopedCapture`) | None |
| **RL framework** | Brax PPO (JAX-native) | RSL-RL PPO (PyTorch) | RSL-RL PPO (PyTorch) via mjlab |
| **Typical num_envs** | 4096 (single GPU kernel) | 16-4096 (GPU tensors, physics throughput varies) | 1-64 (fully CPU-bound physics) |

---

## 1. myosuite `mjx` -- Physics on GPU

### Core Mechanism

The MuJoCo physics solver is compiled into GPU kernels via MJX:

```python
# playground_pose_v0.py
spec = mujoco.MjSpec.from_file(config.model_path.as_posix())
spec = self.preprocess_spec(spec)
self._mj_model = spec.compile()
self._mjx_model = mjx.put_model(self._mj_model, impl="warp")
```

This converts the MuJoCo model to NVIDIA Warp (or XLA) GPU kernels. From this point, **all computation stays on GPU**.

### GPU Computing Details

**Pure functional design** -- `reset()` and `step()` are side-effect-free JAX functions operating on `jax.numpy` arrays in GPU memory. State is immutable; updates use `state.replace(data=new_data, obs=obs, reward=reward, done=done)`.

**Batch vectorization** -- `jax.vmap` (applied internally by Brax's wrapper) maps the scalar step function across 4096 environments as a single GPU kernel launch. No Python loop over environments.

**JIT compilation** -- `jax.jit` traces the entire reset->step->reward->observation pipeline and compiles it to an optimized XLA/CUDA program:
```python
# utils.py
jit_env_reset = jax.jit(eval_env.reset)
jit_env_step = jax.jit(eval_env.step)
jit_policy = jax.jit(make_policy(params, deterministic=True))
```
First call takes seconds (compilation); subsequent calls execute the compiled kernel directly.

**Zero CPU<->GPU sync** during training -- data never leaves GPU between environment step and policy gradient update. The entire training loop (collect rollouts -> compute GAE -> update policy) runs on GPU.

**Key-based PRNG** -- `jax.random.split(rng, 3)` provides deterministic, stateless randomness compatible with GPU parallelism. No mutable global random state.

**Autodifferentiation** -- JAX's autograd works through MJX physics, enabling gradient-based optimization and differentiable simulation.

### Physics Optimizations for GPU Throughput

```python
# Reduce collision overhead
self._mj_model.geom_margin = np.zeros(self._mj_model.geom_margin.shape)
# Fewer solver iterations
self._mj_model.opt.iterations = 6
self._mj_model.opt.ls_iterations = 6
# Disable expensive cylinder collisions
for geom in spec.geoms:
    if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        geom.conaffinity = 0
        geom.contype = 0
# Pre-allocate contact buffers for entire batch
data = make_data(..., nconmax=125 * self._config.num_envs, impl="warp")
```

### Data Flow

```
GPU: mjx.step() -> jp.array obs/reward -> Brax PPO update -> repeat
     ^ never leaves GPU ^
```

### Reward / Observation Computation

All vectorized JAX on GPU, operating on shape `(4096, dim)`:

```python
# Pose environment rewards (4 components)
pose_err = state.info['target_angles'] - data.qpos
pose_dist = jp.linalg.norm(pose_err, axis=-1)
act_mag = jp.linalg.norm(data.act, axis=-1)

pose = pose_dist * -reward_config.angle_reward_weight
act_reg = act_mag * -reward_config.ctrl_cost_weight
bonus = (jp.where(pose_dist < th1, 1., 0.) + jp.where(pose_dist < th2, 1., 0.)) * weight
penalty = -1. * (pose_dist > far_th)
reward = pose + act_reg + bonus + penalty

# Observations
obs = jp.concatenate([data.qpos, data.qvel * dt, data.act, target - data.qpos])
```

Action normalization uses sigmoid: `norm_action = 1.0 / (1.0 + jp.exp(-5.0 * (action - 0.5)))`

### Reset Handling

Vectorized conditional reset across all environments simultaneously:
```python
truncation = jp.where(step_count >= max_steps, 1. - done, 0.)
step_count = jp.where(jp.logical_or(done, truncation), 0, step_count)
targets = jp.where(jp.logical_or(done, truncation),
                   self.generate_target_pose(rng), state.info['targets'])
```

### Environment Registry

| Environment | Task | Config |
|---|---|---|
| `MjxElbowPoseFixed-v0` | 1-DOF elbow, fixed target | num_envs=4096, ctrl_dt=0.02, sim_dt=0.002 |
| `MjxElbowPoseRandom-v0` | 1-DOF elbow, random target | same |
| `MjxFingerPoseFixed-v0` | Multi-joint finger, fixed target | same |
| `MjxFingerPoseRandom-v0` | Multi-joint finger, random target | same |
| `MjxHandReachFixed-v0` | 5-finger hand, fixed reach | same |
| `MjxHandReachRandom-v0` | 5-finger hand, random reach | same |

### Training

```python
from brax.training.agents.ppo import train as ppo
make_inference_fn, params, _ = ppo.train(
    environment=env, num_envs=4096,
    episode_length=max_episode_steps,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    network_factory=make_ppo_networks,  # Flax: 3x64 hidden layers
    **ppo_params)
```

PPO config: 40M timesteps, lr=3e-4, 32 minibatches, 4 update epochs, GAE lambda=0.95, entropy=1e-3, reward_scaling=1.0, 128 eval envs.

### Dependencies
`jax>=0.4.20`, `jaxlib>=0.4.20`, `brax>=0.10.0`, `flax>=0.8.0`, `optax>=0.1.7`, `mujoco==3.3.7`, `ml-collections`, `wandb`, `mediapy`

---

## 2. mjlab -- CPU Physics + GPU Tensor Bridge with CUDA Graphs

### Core Mechanism

MuJoCo's C physics solver runs on CPU. The simulation state is replicated onto GPU via `mujoco_warp` (mjwarp), and all downstream RL computation (observations, rewards, actions) operates on **zero-copy GPU tensors**. **CUDA graph capture** amortizes kernel launch overhead.

```python
# sim.py -- initialization
self._wp_model = mjwarp.put_model(self._mj_model)
self._wp_data = mjwarp.put_data(self._mj_model, self._mj_data,
                                 nworld=self.num_envs, nconmax=cfg.nconmax)
self._model_bridge = WarpBridge(self._wp_model, nworld=self.num_envs)
self._data_bridge = WarpBridge(self._wp_data)
```

### GPU Computing Details

#### Zero-Copy Data Bridge

`TorchArray` wraps Warp GPU arrays as PyTorch tensors **sharing the same GPU memory**:

```python
# sim_data.py
class TorchArray:
    def __init__(self, wp_array):
        self._tensor = wp.to_torch(wp_array)  # shared GPU memory, no copy
        self._torch_stream = ExternalStream(...)  # CUDA stream coordination

    def __getitem__(self, idx):
        return self._tensor[idx]

    def __setitem__(self, idx, val):
        # In-place only -- safe for CUDA graphs
        with torch.cuda.stream(self._torch_stream):
            self._tensor[idx] = val
```

`WarpBridge` provides read-only access to prevent accidental pointer invalidation:

```python
class WarpBridge:
    def __getattr__(self, name):
        val = getattr(self._struct, name)
        if isinstance(val, wp.array):
            return TorchArray(val, nworld=self._nworld)  # cached, zero-copy
        return val

    def __setattr__(self, name, val):
        raise AttributeError(  # CRITICAL: prevents breaking CUDA graphs
            f"Cannot set '{name}' on WarpBridge. Use in-place: obj.{name}[:] = value")
```

#### CUDA Graph Capture

Records GPU kernel sequences for zero-overhead replay:

```python
# sim.py -- graph creation
def create_graph(self):
    with wp.ScopedDevice(self.wp_device):
        with wp.ScopedCapture() as capture:
            mjwarp.step(self.wp_model, self.wp_data)
        self.step_graph = capture.graph

        with wp.ScopedCapture() as capture:
            mjwarp.forward(self.wp_model, self.wp_data)
        self.forward_graph = capture.graph

        with wp.ScopedCapture() as capture:
            mjwarp.reset_data(self.wp_model, self.wp_data, reset=self._reset_mask_wp)
        self.reset_graph = capture.graph

        # Sensing graph: BVH refit + render + raycast kernels
        with wp.ScopedCapture() as capture:
            mjwarp.refit_bvh(self.wp_model, self.wp_data, rc)
            mjwarp.render(self.wp_model, self.wp_data, rc)
            for sensor in ctx.raycast_sensors:
                sensor.raycast_kernel(rc=rc)
        self.sense_graph = capture.graph
```

Replay via single launch:
```python
def step(self):
    with wp.ScopedDevice(self.wp_device):
        if self.use_cuda_graph and self.step_graph is not None:
            wp.capture_launch(self.step_graph)  # replay entire kernel sequence
        else:
            mjwarp.step(self.wp_model, self.wp_data)  # fallback
```

Requirements: CUDA driver >= 12.4, GPU memory pools enabled. Falls back to direct kernel calls on CPU.

**Critical caveat**: CUDA graphs hold pointers to GPU arrays captured at recording time. If arrays are replaced (e.g., by `expand_model_fields()`), the graph silently reads stale memory. Must call `create_graph()` after any array replacement.

#### Model Field Expansion (Domain Randomization)

```python
# randomization.py -- GPU kernel tiles model fields across N worlds
@wp.kernel
def repeat_array_kernel(src, nelems_per_world, dst):
    tid = wp.tid()
    src_idx = tid % nelems_per_world
    dst[tid] = src[src_idx]

def expand_model_fields(model, nworld, fields):
    for field in fields:
        array = getattr(model, field)
        # Allocate new array with first dim = nworld, fill via GPU kernel
        new_array = wp.zeros((nworld, *array.shape[1:]), dtype=array.dtype)
        wp.launch(repeat_array_kernel, ...)
        setattr(model, field, new_array)
```

#### Batched Entity State

`EntityData` maintains PyTorch tensors in shape `(nworld, feature_dim)`:
```
Root State (13D):    position(3) + quaternion(4) + lin_vel(3) + ang_vel(3)
Joint State:         qpos/qvel per environment
Control Targets:     effort/velocity commands
External Wrenches:   forces/torques per body
```

All state writes are in-place for CUDA graph safety:
```python
# CORRECT: in-place modification
entity.data.qpos[env_ids, joint_ids] = new_positions

# WRONG: direct assignment (breaks CUDA graphs)
entity.data.qpos = new_tensor  # raises AttributeError via WarpBridge
```

#### Manager-Based Architecture

Seven composable managers process batched GPU tensors:

**ObservationManager** -- Multi-stage pipeline per term:
```
Compute (GPU) -> Noise -> Clip -> Scale -> Delay Buffer -> History Buffer -> Concatenate
```
Configuration per term: `NoiseCfg`, `clip=(min,max)`, `scale`, `delay_min/max_lag`, `history_length`, `nan_policy`.
Output modes: concatenated tensor or dict of tensors. History uses `CircularBuffer`, delay uses `DelayBuffer`.

**RewardManager** -- Weighted term aggregation:
```python
def compute(self, dt):
    self._reward_buf[:] = 0.0
    scale = dt if self._scale_by_dt else 1.0
    for term_cfg in self._term_cfgs:
        value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * scale
        value = torch.nan_to_num(value, nan=0.0)  # NaN safety
        self._reward_buf += value
        self._episode_sums[name] += value
```
DT scaling normalizes cumulative episode rewards across control frequencies.

**ActionManager** -- Process -> history -> distribute -> apply:
```python
def process_action(self, action):
    self._prev_prev_action[:] = self._prev_action
    self._prev_action[:] = self._action
    self._action[:] = action.to(self.device)
    for term in self._terms.values():
        term.process_actions(action_slice)

def apply_action(self):
    for term in self._terms.values():
        term.apply_actions()  # writes to sim.data.ctrl
```

**EventManager** -- Domain randomization with three modes:
- `"startup"`: once at initialization
- `"reset"`: every episode reset (per-env)
- `"interval"`: periodic, with per-env or global timers

Events tagged `domain_randomization=True` or decorated `@requires_model_fields()` trigger model field expansion.

**TerminationManager** -- Per-environment done flags:
```python
_truncated_buf |= time_limit_value
_terminated_buf |= failure_value
dones = _truncated_buf | _terminated_buf
```

**CurriculumManager** -- Modifies environment parameters during training for progressive difficulty.

**CommandManager** -- Dynamic goal resampling at configurable intervals.

#### GPU-Accelerated Sensors

**Raycast sensor**: BVH-accelerated Warp kernels for terrain height sensing.
- Pre-graph: transform ray origins/directions to world frame via body poses
- In-graph: `mjwarp.refit_bvh()`, then per-sensor `raycast_kernel()` (GPU ray intersection)
- Post-graph: compute hit positions from distances
- Patterns: grid (parallel rays), pinhole camera (diverging rays)
- Geom group filtering, ray alignment modes (base/yaw/world)

**Camera sensor**: `mjwarp.render()` produces ABGR packed uint32; unpacked to RGB via Warp kernel.

**PD Actuator**: Parallel control loop on GPU:
```python
def compute(self, cmd):
    torque = self.stiffness * (cmd.position_target - cmd.pos)
    torque += self.damping * (cmd.velocity_target - cmd.vel)
    torque += cmd.effort_target
    return torch.clamp(torque, -self.force_limit, self.force_limit)
```

#### NaN Guarding
Configurable numerical stability detection: `"disabled"` (fast), `"warn"` (log + sanitize), `"sanitize"` (silent), `"error"` (strict raise).

#### TF32 Precision
Optional reduced-precision mode (10-bit mantissa vs 23-bit FP32): `configure_torch_backends(allow_tf32=True)`.

### Data Flow

```
1. Policy outputs action tensor (GPU)
2. ActionManager distributes to terms -> writes sim.data.ctrl (GPU, zero-copy)
3. Decimation loop:
   a. scene.write_data_to_sim() -- sync entity state
   b. sim.step() -- CUDA graph replay of mjwarp.step()
   c. scene.update(dt) -- entity physics updates
4. sim.forward() -- single CUDA graph replay of mjwarp.forward()
5. TerminationManager / RewardManager -- batched GPU tensor ops
6. Reset terminated envs: sim.reset(env_ids) -- CUDA graph + event_manager("reset")
7. sim.sense() -- CUDA graph: BVH + render + raycast
8. ObservationManager.compute() -- full pipeline on GPU tensors
9. Return (obs, reward, terminated, truncated, extras) -- all GPU tensors
```

### Step Execution (`ManagerBasedRlEnv.step()`)

```python
def step(self, action):
    self.action_manager.process_action(action)
    for _ in range(self.cfg.decimation):
        self.action_manager.apply_action()
        self.scene.write_data_to_sim()
        self.sim.step()                        # CUDA graph or direct mjwarp.step()
        self.scene.update(dt=self.physics_dt)
    self.sim.forward()                         # single forward kinematics pass
    self.termination_manager.compute()         # batched GPU tensor ops
    self.reward_manager.compute(dt)            # batched GPU tensor ops
    self._reset_idx(terminated_envs)           # sim.reset() + event_manager("reset")
    self.sim.forward()                         # refresh kinematics for ALL envs
    self.command_manager.compute()
    self.event_manager.apply(mode="interval")
    self.sim.sense()                           # CUDA graph: BVH + render + raycast
    self.observation_manager.compute()         # full pipeline on GPU
    return obs, reward, terminated, truncated, extras
```

**Design note**: Forward kinematics are computed once after decimation + reset. Non-reset envs have one-substep staleness in derived quantities (xpos, xquat); reset envs get fresh kinematics. This consistent staleness is acceptable for RL.

### Training

```python
# RSL-RL via RslRlVecEnvWrapper -> MjlabOnPolicyRunner
RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(actor_hidden_dims=(256,256), critic_hidden_dims=(256,256)),
    algorithm=RslRlPpoAlgorithmCfg(learning_rate=3e-4, num_learning_epochs=5, num_mini_batches=4),
)
```

`RslRlVecEnvWrapper` adapts `ManagerBasedRlEnv` to RSL-RL: extracts env properties, handles initial reset, optional action clipping, returns `TensorDict`.
`MjlabOnPolicyRunner` extends `OnPolicyRunner`: persists `common_step_counter` for curriculum, migrates legacy checkpoint formats, ONNX export.

### Example Task: Velocity Tracking

Observations: lin/ang velocity, joint states, commands, terrain heights (actor); + foot contacts, height measurements (critic).
Rewards: velocity tracking (2.0 each), upright posture (1.0), joint limit penalty, action rate penalty, foot slip penalty.
Events: state resets, random pushes (1-3s intervals), friction/encoder/COM randomization.
Terminations: 20s timeout, 70 degree orientation limit.
Curriculum: progressive terrain difficulty + velocity range.
Physics: timestep=0.005s, iterations=10, decimation=4.

### Dependencies
`warp-lang>=1.12.0.dev`, `mujoco-warp>=3.5.0`, `mujoco>=3.5.0`, `torch>=2.7.0`, `rsl-rl-lib==4.0.1`, `tensordict`, `viser>=1.0.21`, `wandb>=0.22.3`

---

## 3. mjlab_myosuite -- CPU Physics + GPU Tensor Shuttle (Wrapper)

### Core Mechanism

Standard MuJoCo C library runs physics on CPU. The `MyoSuiteVecEnvWrapper` converts NumPy outputs to PyTorch tensors on the target GPU device via **pinned memory buffers**.

```python
# vec_env_wrapper.py
class MyoSuiteVecEnvWrapper(VecEnv, gym.Env):
    is_vector_env = True
```

### GPU Computing Details

#### Vectorization
```python
if isinstance(env, vector.VectorEnv):
    self.env = env
else:
    # Deep-copy N times, wrap in SyncVectorEnv
    env_factories = [lambda: copy.deepcopy(env) for _ in range(num_envs)]
    self.env = SyncVectorEnv(env_factories)
```
Each environment is an independent CPU process with its own MuJoCo model/data. Steps are sequential.

#### Device Management
```python
def _normalize_device(d):
    d_str = str(d).strip().lower()
    if d_str.startswith("cuda"):
        return torch.device(f"cuda:{idx}" if ":" in d_str else "cuda")
    return torch.device("cpu")

self.device = _normalize_device(device)
```

#### Pinned Memory Transfers (Critical GPU Path)
```python
def _numpy_to_device(self, arr):
    arr = np.asarray(arr, dtype=np.float32)
    if self.device.type != "cuda":
        return torch.from_numpy(arr).to(device=self.device)

    # GPU path: reuse pinned memory buffer
    shape = arr.shape
    if self._pinned_policy_buffer is None or self._pinned_policy_buffer.shape != shape:
        self._pinned_policy_buffer = torch.empty(shape, dtype=torch.float32, pin_memory=True)
    self._pinned_policy_buffer.copy_(torch.from_numpy(arr))
    return self._pinned_policy_buffer.to(device=self.device, non_blocking=True)
```

Optimizations:
- **Pinned memory**: pre-allocated `pin_memory=True` buffer avoids repeated allocation and enables DMA
- **Non-blocking**: `non_blocking=True` overlaps GPU transfer with CPU work
- **Shape caching**: reuses buffer if shape unchanged
- **Contiguous tensors**: `.contiguous()` ensures proper memory layout for downstream ops

#### Step Data Flow
```python
def step(self, actions):
    # GPU -> CPU for physics
    if isinstance(actions, torch.Tensor) and actions.is_cuda:
        actions_np = actions.cpu().numpy()
    else:
        actions_np = actions.numpy()

    # CPU physics
    obs, rew, terminated, truncated, info = self.env.step(actions_np)

    # CPU -> GPU for training
    obs_dict = self._convert_obs_to_dict(obs)  # uses _numpy_to_device()
    rew_tensor = torch.as_tensor(rew, device=self.device, dtype=torch.float32)
    terminated_tensor = torch.as_tensor(terminated, device=self.device, dtype=torch.bool)
    truncated_tensor = torch.as_tensor(truncated, device=self.device, dtype=torch.bool)

    return TensorDict(obs_dict, batch_size=[self.num_envs]), rew_tensor, dones, extras
```

#### Observation Handling

`get_observations()` ensures all tensors are on the correct device with explicit checks:
```python
def get_observations(self):
    obs_dict_on_device = {}
    for key, value in self._last_obs_dict.items():
        if isinstance(value, torch.Tensor):
            if value.device != self.device:
                obs_dict_on_device[key] = value.to(device=self.device).contiguous()
            else:
                obs_dict_on_device[key] = value.contiguous()
        elif isinstance(value, np.ndarray):
            obs_dict_on_device[key] = torch.from_numpy(value).to(
                device=self.device, dtype=torch.float32).contiguous()
    return TensorDict(obs_dict_on_device, batch_size=[self.num_envs])
```

Observations mapped to both `"policy"` and `"critic"` keys for RSL-RL compatibility (no privileged information in MyoSuite).

### Data Flow

```
GPU: RSL-RL policy -> torch action tensor
       | .cpu().numpy() sync
CPU: actions_np -> SyncVectorEnv -> N x mujoco.step() (sequential)
       | numpy obs/rew/done
CPU: _numpy_to_device() with pinned memory buffer
       | pin_memory + non_blocking transfer
GPU: torch TensorDict -> RSL-RL policy update
```

### Compatibility Infrastructure

**MockSimPrimary** (`sim_compat.py`) -- Inherits from `mjlab.sim.sim.Simulation` to pass `isinstance` checks:
```python
class MockSimPrimary(Simulation):
    def __init__(self, env):
        # Don't call super().__init__() -- bypass Warp setup
        self._mj_model = getattr(env, "mj_model", getattr(env, "model", None))
        self._mj_data = getattr(env, "mj_data", getattr(env, "data", None))
        self._wp_data = MockWpData(env, num_envs=1)
        self.num_envs = 1
        self.device = "cpu"
```

**MockWpData** -- Wraps mj_data arrays with batch dimension and `.numpy()` method for Viser viewer:
```python
class MockWpData:
    def _to_batched(self, arr):
        return arr[np.newaxis, :]  # add batch dim

    @property
    def qpos(self):
        return ArrayProxy(self._to_batched(self._mj_data.qpos))
```

**DataAdapter** -- Provides PyTorch tensor interface over mj_data for OffscreenRenderer:
```python
class DataAdapter:
    @property
    def qpos(self):
        return torch.from_numpy(self._mj_data.qpos.copy()).unsqueeze(0)
```

**Mock Managers** (`mocks.py`) -- For ONNX export compatibility:
- `_MockActionManager`: exposes `action_space`, `get_term()` returning action scales
- `_MockObservationManager`: exposes `active_terms` with observation names
- `_MockScene`: provides `mj_model` access

### Environment Factory

```python
def make_myosuite_env(myosuite_env_id, cfg=None, device="cpu", num_envs=None, **kwargs):
    myosuite_gym = _import_myosuite_gym()  # tries mjx paths first
    myosuite_env = myosuite_gym.make(myosuite_env_id, **kwargs)
    return MyoSuiteVecEnvWrapper(env=myosuite_env, num_envs=num_envs, device=device)
```

The factory tries multiple MyoSuite import paths (`myosuite.utils.gym`, `myosuite.utils`, `myosuite`) to support both standard and MJX versions.

### Training

```python
# scripts/train.py
device = f"cuda:{local_rank}"
cfg.env.device = device

env = ManagerBasedRlEnv(cfg=cfg.env, device=device)
runner = MyoSuiteOnPolicyRunner(env, agent_cfg, log_dir, device)
runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
```

`MyoSuiteOnPolicyRunner` extends `OnPolicyRunner` with ONNX export on save:
```python
class MyoSuiteOnPolicyRunner(OnPolicyRunner):
    def save(self, path, infos=None):
        super().save(path, infos)
        export_myosuite_policy_as_onnx(actor_critic=self.alg.policy, ...)
        attach_myosuite_onnx_metadata(env=self.env.unwrapped, ...)
```

### Tracking Environments

Full motion tracking support via `MyoSuiteTrackingEnvCfg` and `make_myosuite_tracking_env()`:
- Loads reference motion files
- Custom reward (tracking error + regularization)
- Specialized runner `MyoSuiteMotionTrackingOnPolicyRunner`

### Dependencies
`mjlab @ git+https://github.com/mujocolab/mjlab.git`, `torch>=2.7.0`, `ruff>=0.14.0`
(inherits mjlab's full dependency chain: mujoco-warp, warp-lang, rsl-rl-lib, tensordict, viser, wandb)

---

## Full Comparison Table

| Dimension | **myosuite `mjx`** | **mjlab** | **mjlab_myosuite** |
|---|---|---|---|
| **Physics location** | GPU (MJX compiled kernels) | CPU (MuJoCo C) -> GPU bridge (mjwarp) | CPU (MuJoCo C) only |
| **GPU data access** | Native JAX arrays on GPU | Zero-copy Warp<->PyTorch (`TorchArray`) | Copy: NumPy -> pinned memory -> GPU tensor |
| **Kernel optimization** | `jax.jit` (full XLA trace) | CUDA graph capture (`wp.ScopedCapture`) | None |
| **Vectorization** | `jax.vmap` -- single model, one kernel | `mjwarp.put_data(nworld=N)` -- N replicated worlds | `SyncVectorEnv` -- N sequential CPU copies |
| **Scalable batch size** | 4096+ on one GPU | 16-4096 (GPU tensors, CPU physics bottleneck) | 1-64 (fully CPU-bound) |
| **CPU<->GPU transfers** | None during training | Minimal (zero-copy bridge) | Every step: obs->GPU, actions->CPU |
| **State management** | Functional immutable (`state.replace()`) | In-place mutation (CUDA graph safe, WarpBridge read-only) | Object-oriented (`self.env.step()`) |
| **Autodiff through physics** | Yes (JAX autograd) | No | No |
| **JIT compilation** | Yes (`jax.jit`) | No (CUDA graphs serve similar purpose) | No |
| **Domain randomization** | Manual per-env (`jp.where`) | `EventManager` + `expand_model_fields()` GPU kernel | Not built-in |
| **Observation pipeline** | Manual `jp.concatenate` | Manager: compute->noise->clip->scale->delay->history | Manual concatenation + pinned transfer |
| **Reward pipeline** | Manual `jp.where` + weights | Manager: weighted terms, dt-scaling, NaN guard | Manual torch computation |
| **Action pipeline** | Sigmoid normalization | Manager: process->history->distribute->apply | Pass-through with optional clipping |
| **Sensor support** | Site positions only | Raycast (BVH/Warp), cameras, contacts | MuJoCo sensor API (CPU) |
| **PRNG** | Key-based (`jax.random.split`) | PyTorch (`torch.manual_seed`) | Seed-based (`np.random`) |
| **Reset handling** | Vectorized `jp.where` conditional | Manager-orchestrated, per-env mask, CUDA graph | `SyncVectorEnv` auto-reset (sequential) |
| **NaN handling** | None explicit | Configurable: disabled/warn/sanitize/error | None |
| **ONNX export** | Not integrated | Native via `MjlabOnPolicyRunner` | Built-in via mock managers + custom exporter |
| **Viewer** | Custom visualization scripts | Viser (web), offscreen, native MuJoCo | Full mjlab stack via mock sim |
| **Curriculum** | Not present | `CurriculumManager` | Not present |
| **Terrain** | Not present | Terrain generation + raycast sensing | Not present |
| **Motion tracking** | Not present | Via command manager | Full tracking task support |
| **Dependencies** | JAX, Brax, Flax, Optax | PyTorch, mujoco_warp, warp-lang, RSL-RL | PyTorch, mjlab (full), MyoSuite |

---

## Similarities Across All Three

1. **MuJoCo as physics foundation** -- All three use MuJoCo XML models, differing only in compilation target (XLA/Warp GPU kernels vs CPU C library vs CPU C library with GPU bridge).
2. **PPO as default RL algorithm** -- Brax PPO (JAX) / RSL-RL PPO (PyTorch) with similar hyperparameters (LR=3e-4, 2-3 hidden layers of 64-256 units, GAE lambda ~0.95).
3. **Observation structure** -- Joint positions, velocities, actuator activations, and task-specific error signals concatenated into flat vectors.
4. **Multi-component rewards** -- Tracking error + control regularization + bonus/penalty thresholds.
5. **Batched environment design** -- All support multiple parallel environments (differ in mechanism and scale).
6. **Headless rendering** -- `MUJOCO_GL=egl` for server environments.
7. **Warp awareness** -- MJX uses `impl="warp"` for kernels; mjlab uses native `mujoco_warp`; mjlab_myosuite provides mock `wp_data` for viewer compatibility.

---

## Key Architectural Differences

### GPU Execution Model

**MJX (JAX/XLA)**: Traces the entire Python function -> compiles to a single fused GPU program -> executes repeatedly. The compiled program includes physics, reward, observation, and reset logic. No Python overhead after first call. Entire training loop (env step + policy update) stays on GPU.

**mjlab (Warp/CUDA Graphs)**: Records sequences of Warp GPU kernel launches -> replays via `wp.capture_launch()`. Four separate graphs: `step_graph`, `forward_graph`, `reset_graph`, `sense_graph`. Manager computations (obs/reward/action) use standard PyTorch tensor ops on GPU but are NOT captured in graphs. Zero-copy `WarpBridge`/`TorchArray` eliminates data movement between Warp and PyTorch.

**mjlab_myosuite (PyTorch copy)**: No GPU kernel compilation or capture. Each step: Python dispatch -> CPU physics -> pinned memory copy -> GPU tensors -> PyTorch ops -> `.cpu().numpy()` copy back. Pinned memory + `non_blocking=True` is the main optimization.

### Composability vs Throughput

**MJX** maximizes throughput but reward/observation/reset logic is hand-coded per environment. Adding a new observation or reward term requires modifying the environment class and re-tracing.

**mjlab** provides a rich manager architecture. Observations, rewards, actions, domain randomization events, terminations, commands, and curriculum are all declaratively configured. Adding a terrain heightmap sensor or friction randomization is a config change. The manager-based design is extensible without touching core environment code.

**mjlab_myosuite** inherits MyoSuite's existing environment logic and wraps it for mjlab/RSL-RL compatibility. Minimal code changes to bring existing MyoSuite environments into mjlab's training, visualization, and ONNX export pipeline.

### Memory Model

**MJX**: Immutable JAX arrays. `state.replace(data=new_data)` creates new state objects. Garbage collected. No aliasing concerns. Fully compatible with XLA tracing.

**mjlab**: In-place mutation on fixed-address GPU arrays. Required by CUDA graph capture (graphs hold pointers to specific memory locations). `WarpBridge.__setattr__` raises `AttributeError` to prevent accidental pointer invalidation. `expand_model_fields()` creates new arrays and requires `create_graph()` to re-capture.

**mjlab_myosuite**: Standard NumPy/PyTorch semantics. No fixed-address constraints. Pinned memory buffers are shape-cached but freely reallocated.

### Staleness Model

**MJX**: No staleness -- JIT-compiled pipeline computes everything in sequence within a single traced function.

**mjlab**: One-substep staleness in derived quantities (xpos, xquat, cvel) after `sim.step()`. A single `sim.forward()` call after the decimation loop + reset refreshes all environments. This consistent staleness is documented and acceptable for RL.

**mjlab_myosuite**: No staleness concern -- CPU MuJoCo `step()` updates all quantities synchronously.

---

## Conclusion

The three projects occupy different points on a throughput<->composability spectrum:

```
Throughput                                                    Composability
    |                                                              |
    myosuite mjx               mjlab                    mjlab_myosuite
    (4096 envs, all-GPU,       (CUDA graphs, zero-copy,  (wrapper, bridges
     JIT-compiled,              manager architecture,     existing MyoSuite
     hand-coded envs)           domain randomization,     envs into mjlab
                                sensors, curriculum)      ecosystem)
```

- **myosuite `mjx`** is optimal for raw GPU simulation throughput on simple tasks with fixed observation/reward structure. The entire training loop runs on GPU with zero synchronization.

- **mjlab** is optimal for building complex RL tasks with rich sensor suites (raycasting, cameras), domain randomization, curriculum learning, and terrain. The manager architecture and CUDA graph capture provide both developer productivity and GPU efficiency. Zero-copy Warp<->PyTorch sharing eliminates data movement overhead.

- **mjlab_myosuite** is optimal for leveraging existing MyoSuite biomechanical environments within mjlab's training, visualization, and ONNX export ecosystem without rewriting them. The pinned memory + non-blocking transfer path provides reasonable GPU utilization for the policy network.

The three are designed to be complementary. mjlab_myosuite's `env_factory.py` already tries MJX import paths first and its `sim_compat.py` handles both `mj_model`/`model` attribute conventions, anticipating convergence where MJX-backed environments can be wrapped in mjlab's manager infrastructure.
