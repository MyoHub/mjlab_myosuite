# myosuite_mjlab

Streamlined entrypoint for training [myosuite](https://github.com/MyoHub/myosuite) musculoskeletal models (MyoLeg, MyoSkeleton, MyoLegTorso) with [mjlab](https://github.com/mujocolab/mjlab). **Standalone:** depends only on public mjlab and myosuite; no fork required.

## What this package does

- Registers musculoskeletal velocity, standing, and balance tasks using **public mjlab** APIs plus a small in-repo MDP extension (`myosuite_mjlab.mdp`) for tendon/synergy observations and curricula.
- Resolves robot assets from the **myosuite** package at runtime.
- Provides `mjlab-train` and `mjlab-play` so you can train and evaluate without modifying mjlab.

## Dependencies

- **[mjlab](https://github.com/mujocolab/mjlab)** (public) — Isaac Lab–style API with MuJoCo-Warp.
- **myosuite** — musculoskeletal assets (XMLs, meshes) at `myosuite/simhive/myo_sim` (standard layout for a full pip install).

## Install

```bash
cd mjlab_myosuite
uv sync
```

## Musculoskeletal task IDs

All of these work with **public mjlab** (no fork):

| Task ID                            | Description                                     |
| ---------------------------------- | ----------------------------------------------- |
| `Mjlab-Velocity-Flat-MyoLeg`       | MyoLeg velocity on flat terrain (tendon effort) |
| `Mjlab-Velocity-Flat-MyoSkeleton`  | MyoSkeleton velocity on flat terrain            |
| `Mjlab-Standing-Flat-MyoSkeleton`  | MyoSkeleton standing (zero velocity command)    |
| `Mjlab-Velocity-Flat-MyoLegsTorso` | MyoLegTorso velocity on flat terrain            |
| `Mjlab-Balance-Flat-MyoLegsTorso`  | MyoLegTorso standing balance (synergy actions)  |

## Train and play

```bash
# Train (example: MyoSkeleton, 4096 envs)
uv run mjlab-train Mjlab-Velocity-Flat-MyoSkeleton --env.scene.num-envs 4096

# Play (e.g. zero agent for sanity check)
uv run mjlab-play Mjlab-Velocity-Flat-MyoSkeleton --agent zero
```

## Assets

Models are resolved from the **myosuite** package. For a full pip install, assets live under **`myosuite/simhive/myo_sim`** (leg, torso, head, etc.). Override paths if needed:

- `MYOSUITE_MJLAB_MYOLEG_XML=/path/to/myolegs_mjlab.xml`
- `MYOSUITE_MJLAB_MYOBODY_XML=/path/to/myobody.xml`
- **MyoLegsTorso / balance:** `MYOSUITE_MJLAB_MYO_SIM` — myo_sim root (must contain `leg/`, `torso/`, `head/`). Only needed if that tree is not at `myosuite/simhive/myo_sim` (e.g. minimal install or custom layout).

## Testing

- **Loadability:** `uv run pytest tests/` runs tests that load registered tasks (no extra setup).

## Adding new models and tasks

### New robot model

1. **Robot config** — Add a `robot_cfg.py` under a task folder (e.g. `src/myosuite_mjlab/tasks/velocity/<robot>/robot_cfg.py`) defining `ROBOT_CFG` (scene entity, XML path resolved from myosuite or env vars).
2. **Env config** — In the same folder, add or extend `env_cfgs.py`: observations, rewards, terminations, and curricula. Use `myosuite_mjlab.mdp` for tendon/synergy helpers (e.g. `tendon_length`, `actuator_force`, `SynergyTendonEffortActionCfg`, `cyclic_hip_flexion_penalty`).
3. **RL config** — Add `rl_cfg.py` with algorithm and experiment settings; expose an env config name that your loader will use.

### New task (velocity / standing / balance)

1. **Package** — Create a directory under `tasks/velocity/`, `tasks/standing/`, or (in mjlab) `tasks/balance/config/`, e.g. `tasks/velocity/<robot>/`.
2. **Loader** — Implement a small loader that builds the env config (e.g. `make_velocity_env_cfg` with your robot and env_cfg), then call `register_mjlab_task(...)` with the task name and config.
3. **Registration** — In that package’s `__init__.py`, import the loader so it runs on `import myosuite_mjlab.tasks`; the task will then appear in `mjlab.tasks.registry.list_tasks()` and be usable with `mjlab-train` / `mjlab-play`.

Use existing tasks as templates: e.g. `tasks/velocity/myoleg/` (robot_cfg, env_cfgs, rl_cfg, `__init__.py` registration).
