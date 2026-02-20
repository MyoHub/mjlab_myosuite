# Analysis: mjlab CPU-GPU Association and What mjlab_myosuite Should Change

## 1. How mjlab Handles CPU-GPU Association

### Key Insight: ONE Environment, ONE Class, Device as Parameter

mjlab does **not** provide two different environments (CPU vs GPU). It provides a **single `ManagerBasedRlEnv` class** where `device` is just a constructor parameter -- exactly like `torch.Tensor(data, device="cuda:0")`. The same code path runs on CPU or GPU:

```python
# Both of these use the SAME class, SAME code:
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")     # CPU physics
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")  # GPU physics
```

### The Internal Architecture

The CPU-GPU transition happens inside `Simulation.__init__`, not at the environment level:

```
ManagerBasedRlEnv.__init__(cfg, device)
  │
  ├── Scene(cfg.scene)
  │     └── .compile() → mujoco.MjModel  [always CPU -- MjModel is a C struct]
  │
  ├── Simulation(num_envs, cfg.sim, model, device)
  │     ├── self._mj_model = model                    # CPU reference (kept for serialization)
  │     ├── self._mj_data  = mujoco.MjData(model)     # CPU reference
  │     │
  │     ├── with wp.ScopedDevice(device):              # <-- THIS is the CPU/GPU switch
  │     │     self._wp_model = mjwarp.put_model(model)
  │     │     self._wp_data  = mjwarp.put_data(model, data, nworld=num_envs)
  │     │
  │     ├── self.model = WarpBridge(wp_model)          # Zero-copy → PyTorch tensors
  │     └── self.data  = WarpBridge(wp_data)           # Zero-copy → PyTorch tensors
  │
  ├── ObservationManager(cfg.observations, env=self)   # reads env.sim.data.*
  ├── ActionManager(cfg.actions, env=self)             # writes env.sim.data.ctrl
  ├── RewardManager(cfg.rewards, env=self)             # reads env.sim.data.*
  ├── TerminationManager(cfg.terminations, env=self)
  ├── EventManager(cfg.events, env=self)
  └── CommandManager(cfg.commands, env=self)
```

**The critical pattern:**
1. `MjModel` is always compiled on CPU (it's a C struct)
2. `Simulation` receives a device string and uses `wp.ScopedDevice(device)` to control where Warp arrays are allocated
3. When `device="cuda:0"`, `mjwarp.put_model()` copies the model to GPU, `mjwarp.put_data(nworld=N)` creates N parallel data instances on GPU
4. When `device="cpu"`, the same calls allocate on CPU
5. `WarpBridge` wraps Warp arrays and provides zero-copy `TorchArray` views -- the tensors are on whatever device Warp allocated them on
6. All managers receive `env=self` and access `env.sim.data.*` -- they never check the device, they just operate on tensors

### What Managers See

Manager terms are device-agnostic functions:

```python
# This observation term works identically on CPU and GPU:
def joint_positions(env) -> torch.Tensor:
    return env.sim.data.qpos[:, :]  # TorchArray → torch.Tensor (on env.device)
```

The tensor is already on the right device because `WarpBridge`/`TorchArray` wraps a Warp array that was allocated on `device`. No `.to()` calls, no numpy conversion, no pinned memory shuttle.

### CUDA Graphs: Optional Optimization, Not Architecture

CUDA graph capture (`wp.ScopedCapture`) is a conditional optimization:
- If device is CUDA and driver ≥ 12.4 and memory pools enabled → capture graphs
- Otherwise → eager execution
- The step/forward/reset/sense API is the same either way

---

## 2. What mjlab_myosuite Currently Does (and Why It's Different)

### Current Architecture: Two Separate Worlds

mjlab_myosuite currently creates a **parallel universe** that mimics mjlab's interface:

```
                          mjlab native path                    mjlab_myosuite path
                          ─────────────────                    ──────────────────────
Config:                   ManagerBasedRlEnvCfg                 MyoSuiteEnvCfg (stub dataclass)
Scene:                    Scene → MjSpec → compile()           _MockSceneCfg (empty dict)
Simulation:               Simulation(model, device)            MockSimPrimary (no GPU, no mjwarp)
Physics:                  mjwarp.step() on GPU                 MyoSuite.step() on CPU
Managers:                 Real managers with GPU terms          Mock managers (no-ops)
Observations:             ObservationManager.compute()         _convert_obs_to_dict() + pinned memory
Actions:                  ActionManager.apply_actions()         numpy → MyoSuite.step()
Vectorization:            mjwarp.put_data(nworld=N)            SyncVectorEnv(deepcopy × N)
ManagerBasedRlEnv:        Normal __init__                      Monkey-patched __init__ + step + get_obs
```

### Why Mocks Exist

The mocks exist because mjlab_myosuite **does not use mjlab's Simulation at all**. It:
1. Creates MyoSuite envs via `myosuite_gym.make()` (CPU gymnasium environments)
2. Wraps them in `MyoSuiteVecEnvWrapper` (handles numpy→torch, CPU→GPU transfer)
3. Creates `MockSimPrimary` that inherits from `mjlab.sim.Simulation` but **skips `__init__`** (no `mjwarp.put_model`, no GPU data)
4. Creates mock managers that satisfy interface checks but do nothing

The patched `ManagerBasedRlEnv.__init__` detects `MyoSuiteEnvCfg`, creates the CPU environment, stores it, overrides `get_observations()` and `step()` to delegate to the wrapper, and returns. The real mjlab managers are created but never used for actual computation.

### Dependency Map

```
mjlab_myosuite currently USES from myosuite:
  ├── myosuite.utils.gym.make()         # Create CPU environment
  ├── env.step() / env.reset()          # CPU physics + reward + obs
  ├── env.mj_model / env.mj_data       # For viewer compatibility
  └── (everything else is inside MyoSuite)

mjlab_myosuite currently USES from mjlab:
  ├── ManagerBasedRlEnv                 # Patched, used as shell
  ├── mjlab.sim.sim.Simulation          # Base class for MockSim (isinstance checks)
  ├── mjlab.managers.*                  # Base classes for term configs (not used for compute)
  ├── mjlab.rl.RslRlVecEnvWrapper      # Training loop wrapper
  ├── mjlab.viewer.*                    # Rendering
  ├── mjlab.utils.*                     # Configuration, logging
  └── mjlab.scripts.train/play          # Entry points (patched)
```

---

## 3. What Should Change: Minimal-Dependency Unified Interface

### Design Principle: Reuse mjlab's `Simulation` Directly with MyoSuite's `MjModel`

The key realization is that mjlab's `Simulation` class doesn't care **where** the `MjModel` came from. It just needs a compiled `mujoco.MjModel`:

```python
# mjlab native:
model = scene.compile()                       # MjModel from MjSpec
sim = Simulation(num_envs, cfg, model, device) # → GPU or CPU

# What mjlab_myosuite SHOULD do:
model = mujoco.MjModel.from_xml_path(xml)     # MjModel from MyoSuite XML
sim = Simulation(num_envs, cfg, model, device) # → SAME Simulation, SAME GPU path
```

No mocks needed. No patches needed. The `Simulation` class handles CPU/GPU transparently.

### What MyoSuite Provides That We Reuse

From MyoSuite, we need exactly **two things**:
1. **XML model paths** -- extractable from gym registry: `gym.spec("myoElbowPose1D6MRandom-v0").kwargs["model_path"]`
2. **Task logic reference** -- reward functions, observation definitions, termination conditions, reset randomization (these are reimplemented as mjlab manager terms, but MyoSuite is the specification)

We do **NOT** need:
- `myosuite_gym.make()` -- we load the XML directly
- `env.step()` -- mjlab's Simulation handles physics
- `SyncVectorEnv` -- mjwarp handles vectorization
- `MockSimPrimary` -- we use the real Simulation
- Mock managers -- we use real manager terms

### What mjlab Provides That We Reuse

From mjlab, we reuse **everything**:
- `Simulation` -- unmodified, handles CPU/GPU via device parameter
- `ManagerBasedRlEnv` -- unmodified, no patching
- `Scene` -- minimal subclass to load MyoSuite XML instead of MjSpec builder
- Manager infrastructure -- real `ObservationManager`, `ActionManager`, `RewardManager`, etc.
- `RslRlVecEnvWrapper` -- unmodified
- Training/play scripts -- unmodified (tasks registered normally)
- Viewers -- unmodified (Scene provides mj_model)

### Architectural Changes

#### A. Replace `MyoSuiteEnvCfg` (stub) with a real `ManagerBasedRlEnvCfg` subclass

**Current:** `MyoSuiteEnvCfg` is a standalone dataclass with stub fields that mimic `ManagerBasedRlEnvCfg`
**New:** Subclass `ManagerBasedRlEnvCfg` properly, providing:
- A `SceneCfg` subclass that knows how to load MyoSuite XML
- Real observation/action/reward/termination/event term configs
- SimulationCfg with sensible defaults for MyoSuite models

```python
@dataclass
class MyoSuiteSceneCfg(SceneCfg):
    """Scene that loads a MyoSuite XML model."""
    myosuite_env_id: str = ""     # e.g. "myoElbowPose1D6MRandom-v0"
    # entities, sensors, terrain remain empty -- model is self-contained

@dataclass
class MyoElbowPoseEnvCfg(ManagerBasedRlEnvCfg):
    scene: MyoSuiteSceneCfg = MyoSuiteSceneCfg(myosuite_env_id="myoElbowPose1D6MRandom-v0")
    sim: SimulationCfg = SimulationCfg()
    decimation: int = 5
    observations = {"policy": ObservationGroupCfg(terms={...})}
    actions = {"muscles": MyoMuscleActionTermCfg()}
    rewards = {"pose_tracking": ElbowPoseRewardTermCfg(weight=1.0), ...}
    terminations = {"time_limit": TimeLimitTermCfg()}
    events = {"reset_state": RandomizeStateEventCfg(), ...}
```

#### B. Create a minimal `MyoSuiteScene` that loads XML → MjModel

**Current:** `_MockSceneCfg` with empty entities, `MockSimPrimary` bypasses Simulation
**New:** Subclass or adapter for mjlab's `Scene` that overrides `compile()`:

```python
class MyoSuiteScene(Scene):
    def compile(self) -> mujoco.MjModel:
        xml_path = get_xml_path(self.cfg.myosuite_env_id)  # From gym registry
        return mujoco.MjModel.from_xml_path(xml_path)

    # write_data_to_sim, update, reset: either delegate to base or no-op
    # (MyoSuite models are self-contained, no entity system needed)
```

This is the **only** place MyoSuite is accessed. Everything downstream uses the standard `MjModel`.

#### C. Implement task logic as GPU-native manager terms

**Current:** MyoSuite's `env.step()` computes obs/reward/done on CPU, wrapper shuttles to GPU
**New:** Manager terms read from `env.sim.data` (GPU tensors via WarpBridge):

```python
# Observation: reads GPU sim state directly
def myosuite_joint_obs(env) -> torch.Tensor:
    qpos = env.sim.data.qpos[:, :]    # (num_envs, nq) on device
    qvel = env.sim.data.qvel[:, :]    # (num_envs, nv) on device
    act  = env.sim.data.act[:, :]     # (num_envs, na) on device
    return torch.cat([qpos, qvel, act], dim=-1)

# Action: writes GPU ctrl directly
def myosuite_muscle_action(env, action: torch.Tensor):
    env.sim.data.ctrl[:] = action     # In-place for CUDA graph safety

# Reward: computed on GPU tensors
def elbow_pose_reward(env) -> torch.Tensor:
    tip = env.sim.data.xpos[:, fingertip_id]  # (num_envs, 3)
    error = torch.norm(tip - env.target, dim=-1)
    return torch.exp(-error / 0.1)
```

These are standard mjlab manager terms. They work on CPU or GPU depending on where `Simulation` allocated data.

#### D. Remove monkey-patching of `ManagerBasedRlEnv.__init__`

**Current:** `play.py` replaces `ManagerBasedRlEnv.__init__` with `_patched_manager_init` that detects `MyoSuiteEnvCfg` and creates the wrapper
**New:** No patching needed. `ManagerBasedRlEnv.__init__` works unmodified because:
- `cfg` is a proper `ManagerBasedRlEnvCfg` subclass
- `Scene.compile()` returns a valid `MjModel`
- `Simulation` creates GPU data normally
- Managers are real manager terms, not mocks

#### E. Remove all mocks

**Current:** `MockSimPrimary`, `MockWpData`, `DataAdapter`, `_MockScene`, `_MockActionManager`, `_MockObservationManager`, `_MockCommandManager`
**New:** None of these are needed. The real mjlab classes handle everything.

#### F. Keep CPU fallback via device parameter (not separate code path)

**Current:** CPU wrapper is a completely different code path (SyncVectorEnv + pinned memory)
**New:** Same code, `device="cpu"`:

```python
# GPU training (4096 parallel envs on GPU):
env = ManagerBasedRlEnv(cfg, device="cuda:0")

# CPU debugging (1 env on CPU):
env = ManagerBasedRlEnv(cfg, device="cpu")
```

Both use the same `MyoSuiteScene`, same `Simulation`, same managers. The device parameter flows through `wp.ScopedDevice()` in Simulation, and all tensors end up on the right device automatically.

---

## 4. Summary of Changes

### Files to Create

| File | Purpose | Dependencies |
|------|---------|-------------|
| `scene.py` | `MyoSuiteScene` -- loads XML → MjModel | myosuite (gym registry for XML path), mjlab (Scene base) |
| `gpu_env_cfg.py` | Per-task `ManagerBasedRlEnvCfg` subclasses | mjlab (cfg base classes) |
| `gpu_managers.py` | Observation/action/reward/termination/event terms | mjlab (manager term base classes) |

### Files to Modify

| File | Change |
|------|--------|
| `registration.py` | Register tasks with real `ManagerBasedRlEnvCfg` subclasses |
| `scripts/train.py` | Remove `_patched_run_train`; use mjlab's `run_train` directly |
| `scripts/play.py` | Remove `_patched_manager_init`; use mjlab's `run_play` directly |

### Files to Eventually Remove (or Keep Only for Legacy)

| File | Why |
|------|-----|
| `wrapper/sim_compat.py` | No mocks needed; real Simulation used |
| `wrapper/mocks.py` | No mock managers needed; real managers used |
| `wrapper/vec_env_wrapper.py` | No CPU wrapper needed; Simulation handles vectorization |
| `config.py` (MyoSuiteEnvCfg) | Replaced by proper ManagerBasedRlEnvCfg subclasses |

### Dependency Reduction

```
BEFORE (mjlab_myosuite dependencies):
  myosuite:  gym.make(), env.step(), env.reset(), env.mj_model, env.mj_data
  mjlab:     Patched ManagerBasedRlEnv, mock Simulation, mock managers, viewers, rl, utils

AFTER (mjlab_myosuite dependencies):
  myosuite:  gym.spec().kwargs["model_path"] (XML path extraction only)
  mjlab:     ManagerBasedRlEnv (unmodified), Simulation (unmodified), Scene (subclassed),
             manager base classes (subclassed), rl, viewers, utils -- all used as intended
```

MyoSuite dependency reduces to **one function call** (extracting the XML path).
mjlab dependency changes from **patching/mocking** to **normal subclassing/composition**.

---

## 5. Implementation Strategy

### Phase 1: Single Reference Task (myoElbowPose1D6MRandom)
1. Create `MyoSuiteScene` that loads elbow XML via `mujoco.MjModel.from_xml_path()`
2. Create `MyoElbowPoseEnvCfg(ManagerBasedRlEnvCfg)` with real scene/sim/manager configs
3. Implement observation/action/reward/termination terms for this specific task
4. Register as a standard mjlab task
5. Verify: `ManagerBasedRlEnv(cfg, device="cuda:0")` creates GPU physics, runs step cycle

### Phase 2: Generalize to Task Registry
1. Create `MyoSuiteTaskSpec` dataclass mapping task ID → XML path + obs/reward/term configs
2. Create a factory that generates `ManagerBasedRlEnvCfg` for any registered MyoSuite task
3. Port additional tasks (hand, leg, etc.)

### Phase 3: Remove Legacy Code
1. Remove monkey-patching from play.py/train.py
2. Remove mocks (sim_compat.py, mocks.py)
3. Remove MyoSuiteVecEnvWrapper (or keep as opt-in legacy mode)
4. Update registration to use real configs exclusively
