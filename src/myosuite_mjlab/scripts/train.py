"""Standalone training script for MyoSuite-mjlab tasks.

Uses only public mjlab API plus myosuite_mjlab's compatibility layer (step
adapter, agent_cfg adaptation, MyosuiteVelocityRunner). Does not call
mjlab.scripts.train.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

# Set OpenGL backend before any import that loads mujoco.
if not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "egl"
if not os.environ.get("MUJOCO_EGL_DEVICE_ID"):
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

import myosuite_mjlab.tasks  # noqa: F401  # register tasks before list_tasks

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import resolve_curriculum_iterations
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags

from myosuite_mjlab.rl import (
    RslRlStepAdapter,
    adapt_agent_cfg_for_rsl_rl,
)


@dataclass
class TrainConfig:
    """Training configuration."""

    env: ManagerBasedRlEnvCfg
    agent: RslRlOnPolicyRunnerCfg
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    wandb_run_path: str | None = None
    gpu_ids: list[int] | Literal["all"] | None = None

    def __post_init__(self) -> None:
        if self.gpu_ids is None:
            object.__setattr__(self, "gpu_ids", [0])

    @staticmethod
    def from_task(task_id: str) -> TrainConfig:
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def _run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        device = "cpu"
        seed = cfg.agent.seed
        rank = 0
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        seed = cfg.agent.seed + local_rank

    configure_torch_backends()

    cfg.agent.seed = seed
    cfg.env.seed = seed

    print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

    resolve_curriculum_iterations(cfg.env.curriculum, cfg.agent.num_steps_per_env)

    if rank == 0:
        print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = ManagerBasedRlEnv(
        cfg=cfg.env,
        device=device,
        render_mode="rgb_array" if cfg.video else None,
    )

    if cfg.video and rank == 0:
        try:
            from mjlab.utils.wrappers import VideoRecorder

            env = VideoRecorder(
                env,
                video_folder=Path(log_dir) / "videos" / "train",
                step_trigger=lambda step: step % cfg.video_interval == 0,
                video_length=cfg.video_length,
                disable_logger=True,
                log_to_wandb=(cfg.agent.logger == "wandb"),
            )
            print("[INFO] Recording videos during training.")
        except ImportError:
            pass

    vec_env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    vec_env = RslRlStepAdapter(vec_env)

    agent_cfg = asdict(cfg.agent)
    adapt_agent_cfg_for_rsl_rl(agent_cfg)

    resume_path: Path | None = None
    if getattr(cfg.agent, "resume", False):
        log_root_path = log_dir.parent
        if cfg.wandb_run_path:
            resume_path, _ = get_wandb_checkpoint_path(
                log_root_path, Path(cfg.wandb_run_path)
            )
        else:
            resume_path = get_checkpoint_path(
                log_root_path,
                getattr(cfg.agent, "load_run", ".*"),
                getattr(cfg.agent, "load_checkpoint", ".*"),
            )

    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
        runner_cls = MjlabOnPolicyRunner

    runner = runner_cls(vec_env, agent_cfg, str(log_dir), device)

    add_wandb_tags(getattr(cfg.agent, "wandb_tags", []))
    if hasattr(runner, "add_git_repo_to_log"):
        try:
            runner.add_git_repo_to_log(__file__)
        except Exception:
            pass

    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))

    if rank == 0:
        env_cfg_dict = asdict(cfg.env)
        dump_yaml(log_dir / "params" / "env.yaml", env_cfg_dict)
        dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

    try:
        runner.learn(
            num_learning_iterations=cfg.agent.max_iterations,
            init_at_random_ep_len=True,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user (Ctrl+C).", flush=True)
        vec_env.close()
        sys.exit(130)

    vec_env.close()


def _launch_training(task_id: str, args: TrainConfig) -> None:
    log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
    log_root_path.resolve()
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if getattr(args.agent, "run_name", None):
        log_dir_name += f"_{args.agent.run_name}"
    log_dir = log_root_path / log_dir_name

    selected_gpus, num_gpus = select_gpus(args.gpu_ids)

    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_GL"] = "egl"

    if num_gpus <= 1:
        _run_train(task_id, args, log_dir)
    else:
        try:
            import torchrunx
        except ImportError:
            print("[WARN] Multi-GPU requires torchrunx; running single process.")
            _run_train(task_id, args, log_dir)
            return
        logging.basicConfig(level=logging.INFO)
        print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
        torchrunx.Launcher(
            hostnames=["localhost"],
            workers_per_host=num_gpus,
            backend=None,
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
        ).run(_run_train, task_id, args, log_dir)


def _parse_args() -> tuple[str, TrainConfig]:
    all_tasks = list_tasks()
    parser = argparse.ArgumentParser(
        description="Train MyoSuite-mjlab tasks (standalone, public mjlab only)."
    )
    parser.add_argument(
        "task",
        choices=all_tasks,
        help="Task ID to train",
    )
    parser.add_argument(
        "--agent.max-iterations",
        type=int,
        dest="max_iterations",
        metavar="N",
        help="Max training iterations",
    )
    parser.add_argument(
        "--env.scene.num-envs",
        type=int,
        dest="num_envs",
        metavar="N",
        help="Number of parallel envs",
    )
    parser.add_argument(
        "--agent.logger",
        type=str,
        dest="logger",
        choices=("tensorboard", "wandb"),
        help="Logger type",
    )
    parser.add_argument(
        "--video",
        type=lambda x: x.lower() == "true",
        default=False,
        metavar="True|False",
        help="Record videos during training",
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=200,
        help="Frames per video",
    )
    parser.add_argument(
        "--video-interval",
        type=int,
        default=2000,
        help="Record every N steps",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
        help="Comma-separated GPU indices or 'all'",
    )
    args = parser.parse_args()

    cfg = TrainConfig.from_task(args.task)
    if args.max_iterations is not None:
        cfg.agent.max_iterations = args.max_iterations
    if args.num_envs is not None:
        cfg.env.scene.num_envs = args.num_envs
    if args.logger is not None:
        cfg.agent.logger = args.logger
    cfg.video = args.video
    cfg.video_length = args.video_length
    cfg.video_interval = args.video_interval
    if args.gpu_ids.lower() == "all":
        cfg.gpu_ids = "all"
    else:
        cfg.gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    return args.task, cfg


def main() -> None:
    task_id, args = _parse_args()
    _launch_training(task_id=task_id, args=args)


if __name__ == "__main__":
    main()
