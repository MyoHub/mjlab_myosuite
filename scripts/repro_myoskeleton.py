"""Reproduction script for MyoSkeleton task."""

import sys
from pathlib import Path

# Setup paths
src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_path))
mjlab_src = Path(__file__).resolve().parents[2] / "mjlab" / "src"
if mjlab_src.exists():
    sys.path.insert(0, str(mjlab_src))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402


def main():
    task = "Mjlab-Velocity-Flat-MyoSkeleton"
    device = "cpu"

    print(f"Loading task {task}...")
    env_cfg = load_env_cfg(task)
    env_cfg.num_envs = 1
    env_cfg.device = device

    rl_cfg = load_rl_cfg(task)
    rl_cfg.max_iterations = 5

    print("Building environment...")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    vec_env = RslRlVecEnvWrapper(env)

    runner_cls = load_runner_cls(task)
    print(f"Runner class: {runner_cls}")

    print("Run completed successfully (dry run).")
    vec_env.close()


if __name__ == "__main__":
    main()
