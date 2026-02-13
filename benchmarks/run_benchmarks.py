"""
Benchmark suite for mjlab_myosuite: throughput, GPU scaling, and mjlab comparison.

Verifies that mjlab_myosuite and (when available) native mjlab / myosuite-mjx
use GPUs and scale similarly with num_envs.

With --device cuda:0, GPU peak memory is reported. With standard MyoSuite (CPU physics),
only observation and reward tensors are on GPU, so usage is small (often <1 MB) and
nvidia-smi may show little change. For large GPU memory use, install MyoSuite from the
mjx branch and use --with-warp (GPU simulation).

Usage:
  uv run python benchmarks/run_benchmarks.py
  uv run python benchmarks/run_benchmarks.py --device cuda:0 --num-envs 64 256 1024
  uv run python benchmarks/run_benchmarks.py --compare-mjlab --num-envs 64 256 1024
  uv run python benchmarks/run_benchmarks.py --verify-scaling  # assert scaling is similar
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _has_myosuite() -> bool:
  try:
    import myosuite  # noqa: F401

    return True
  except Exception:
    return False


def _has_cuda() -> bool:
  try:
    import torch

    return torch.cuda.is_available()
  except Exception:
    return False


def _gpu_memory_mb() -> tuple[float, float] | None:
  """Return (allocated_mb, reserved_mb) or None if no CUDA."""
  try:
    import torch

    if not torch.cuda.is_available():
      return None
    dev = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(dev) / (1024**2)
    reserved = torch.cuda.memory_reserved(dev) / (1024**2)
    return (round(alloc, 2), round(reserved, 2))
  except Exception:
    return None


def run_throughput(
  task_id: str = "myoElbowPose1D6MRandom-v0",
  num_envs: int = 256,
  device: str = "cpu",
  num_steps: int = 300,
  warmup_steps: int = 30,
  physics_backend: str | None = None,
  report_gpu_memory: bool = True,
) -> dict:
  """Measure steps/sec for mjlab_myosuite env."""
  from mjlab_myosuite.config import MyoSuiteEnvCfg, PhysicsBackend
  from mjlab_myosuite.env_factory import make_myosuite_env

  if device != "cpu" and report_gpu_memory:
    try:
      import torch

      if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
      pass

  mem_before = _gpu_memory_mb() if device != "cpu" else None
  cfg = MyoSuiteEnvCfg()
  cfg.num_envs = num_envs
  cfg.device = device
  if physics_backend == "warp":
    cfg.physics_backend = PhysicsBackend.WARP
  env = make_myosuite_env(task_id, cfg=cfg, num_envs=num_envs, device=device)
  mem_after_create = _gpu_memory_mb() if device != "cpu" else None

  try:
    import numpy as np

    env.reset(seed=42)
    n = env.num_envs
    action_dim = getattr(env.unwrapped, "num_actions", None)
    if action_dim is None and hasattr(env, "single_action_space"):
      sp = env.single_action_space
      action_dim = int(np.prod(getattr(sp, "shape", (1,))))
    else:
      action_dim = action_dim or 1
    for _ in range(warmup_steps):
      env.step(np.zeros((n, action_dim), dtype=np.float32))
    env.reset(seed=123)
    t0 = time.perf_counter()
    for _ in range(num_steps):
      env.step(np.zeros((n, action_dim), dtype=np.float32))
    if device != "cpu":
      import torch

      torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
  finally:
    env.close()

  mem_after = _gpu_memory_mb() if device != "cpu" else None
  peak_mb = None
  if device != "cpu":
    try:
      import torch

      if torch.cuda.is_available():
        peak_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 2)
    except Exception:
      pass

  total = num_steps * n
  out = {
    "backend": "mjlab_myosuite",
    "task_id": task_id,
    "num_envs": n,
    "device": device,
    "physics_backend": physics_backend or "cpu",
    "steps_per_sec": round(total / elapsed, 1),
    "total_steps": total,
    "elapsed_s": round(elapsed, 3),
  }
  if mem_before is not None:
    out["gpu_mem_allocated_before_mb"] = mem_before[0]
    out["gpu_mem_reserved_before_mb"] = mem_before[1]
  if mem_after_create is not None:
    out["gpu_mem_allocated_after_create_mb"] = mem_after_create[0]
  if mem_after is not None:
    out["gpu_mem_allocated_after_mb"] = mem_after[0]
    out["gpu_mem_reserved_after_mb"] = mem_after[1]
  if peak_mb is not None:
    out["gpu_mem_peak_mb"] = peak_mb
  return out


def run_native_mjlab_throughput(
  mjlab_task_id: str,
  num_envs: int,
  device: str = "cuda:0",
  num_steps: int = 300,
  warmup_steps: int = 30,
) -> dict | None:
  """Measure steps/sec for native mjlab env on GPU. Returns None on failure."""
  if device == "cpu":
    return None
  try:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
  except ImportError:
    return None
  if not torch.cuda.is_available():
    return None
  try:
    cfg = load_env_cfg(mjlab_task_id, play=False)
    if (
      hasattr(cfg, "scene") and cfg.scene is not None and hasattr(cfg.scene, "num_envs")
    ):
      object.__setattr__(cfg.scene, "num_envs", num_envs)
    env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  except Exception:
    return None
  try:
    n = env.num_envs
    action_dim = env.single_action_space.shape[0]
    env.reset(seed=42)
    for _ in range(warmup_steps):
      env.step(torch.zeros((n, action_dim), device=device, dtype=torch.float32))
    env.reset(seed=123)
    t0 = time.perf_counter()
    for _ in range(num_steps):
      env.step(torch.zeros((n, action_dim), device=device, dtype=torch.float32))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
  finally:
    env.close()
  total = num_steps * n
  return {
    "backend": "mjlab",
    "task_id": mjlab_task_id,
    "num_envs": n,
    "device": device,
    "steps_per_sec": round(total / elapsed, 1),
    "total_steps": total,
    "elapsed_s": round(elapsed, 3),
  }


def _get_native_mjlab_task() -> str | None:
  """First non-MyoSuite task from mjlab registry."""
  try:
    from mjlab.tasks.registry import list_tasks

    for _ in ("mjlab.tasks.velocity", "mjlab.tasks", "mjlab"):
      try:
        __import__(_)
        break
      except Exception:
        continue
    tasks = list_tasks()
    for t in tasks:
      if not t.startswith("Mjlab-MyoSuite"):
        return t
  except Exception:
    pass
  return None


def main() -> int:
  parser = argparse.ArgumentParser(description="Run mjlab_myosuite benchmarks")
  parser.add_argument(
    "--task", default="myoElbowPose1D6MRandom-v0", help="MyoSuite task ID"
  )
  parser.add_argument("--num-envs", type=int, nargs="+", default=[64, 256, 1024])
  parser.add_argument("--device", default="cpu", help="Device (cpu or cuda:0)")
  parser.add_argument("--steps", type=int, default=300)
  parser.add_argument(
    "--compare-mjlab",
    action="store_true",
    help="Compare mjlab_myosuite GPU vs native mjlab GPU scaling",
  )
  parser.add_argument(
    "--with-warp",
    action="store_true",
    help="Use PhysicsBackend.WARP for mjlab_myosuite when available",
  )
  parser.add_argument(
    "--verify-scaling",
    action="store_true",
    help="Assert that steps/sec scales with num_envs (monotonic or similar)",
  )
  parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
  args = parser.parse_args()

  if not _has_myosuite():
    print("MyoSuite not installed. Skipping benchmarks.")
    return 1

  args.output_dir.mkdir(parents=True, exist_ok=True)
  results: dict = {"throughput": [], "mjlab_comparison": [], "scaling_ok": None}

  # ---- Throughput (single backend) ----
  if not args.compare_mjlab:
    print(f"--- Throughput (device={args.device}) ---")
    for num_envs in args.num_envs:
      print(f"  num_envs={num_envs} ...", end=" ", flush=True)
      r = run_throughput(
        task_id=args.task,
        num_envs=num_envs,
        device=args.device,
        num_steps=args.steps,
        physics_backend="warp" if args.with_warp else None,
      )
      results["throughput"].append(r)
      msg = f"{r['steps_per_sec']:.0f} steps/s"
      if r.get("gpu_mem_peak_mb") is not None:
        msg += f"  (GPU peak: {r['gpu_mem_peak_mb']} MB)"
      print(msg)
    if args.device != "cpu" and results["throughput"]:
      peak = (results["throughput"][0].get("gpu_mem_peak_mb") or 0) or 0.01
      print(
        f"  Note: With standard MyoSuite (CPU physics), only obs/reward tensors are on GPU (peak ~{peak:.2f} MB). "
        "nvidia-smi may show little change. For large GPU use, install MyoSuite mjx branch and use --with-warp."
      )
    print()

  # ---- Compare mjlab vs mjlab_myosuite on GPU ----
  if args.compare_mjlab:
    if not _has_cuda():
      print("--- GPU comparison ---\n  No CUDA. Skipping --compare-mjlab.")
      print()
    else:
      native_task = _get_native_mjlab_task()
      print("--- GPU: mjlab_myosuite vs mjlab scaling ---")
      print(f"  mjlab task: {native_task or 'N/A'}")
      print(f"  mjlab_myosuite task: {args.task}")
      mjlab_sps_list: list[float] = []
      myosuite_sps_list: list[float] = []
      for num_envs in args.num_envs:
        print(f"  num_envs={num_envs}:", end="", flush=True)
        # mjlab_myosuite on GPU
        r_wrap = run_throughput(
          task_id=args.task,
          num_envs=num_envs,
          device="cuda:0",
          num_steps=args.steps,
          physics_backend="warp" if args.with_warp else None,
        )
        results["mjlab_comparison"].append(r_wrap)
        myosuite_sps_list.append(r_wrap["steps_per_sec"])
        gpu_mem = (
          f" (GPU peak: {r_wrap['gpu_mem_peak_mb']} MB)"
          if r_wrap.get("gpu_mem_peak_mb") is not None
          else ""
        )
        print(
          f" mjlab_myosuite={r_wrap['steps_per_sec']:.0f}{gpu_mem}", end="", flush=True
        )
        # Native mjlab on GPU
        if native_task:
          r_native = run_native_mjlab_throughput(
            native_task, num_envs, "cuda:0", num_steps=args.steps
          )
          if r_native is not None:
            results["mjlab_comparison"].append(r_native)
            mjlab_sps_list.append(r_native["steps_per_sec"])
            print(f" mjlab={r_native['steps_per_sec']:.0f}", end="", flush=True)
          else:
            print(" mjlab=N/A", end="", flush=True)
        print(" steps/s")
      # Scaling check: both should scale (steps/sec non-decreasing or similar trend)
      if args.verify_scaling and len(myosuite_sps_list) >= 2:
        # mjlab_myosuite: steps/sec should generally increase with num_envs (or stay similar)
        scaling_ok = myosuite_sps_list[-1] >= 0.5 * myosuite_sps_list[0]
        results["scaling_ok"] = scaling_ok
        if not scaling_ok:
          print(
            "  [FAIL] mjlab_myosuite scaling: steps/sec did not scale with num_envs."
          )
        else:
          print("  [OK] mjlab_myosuite scales with num_envs.")
        if mjlab_sps_list and len(mjlab_sps_list) >= 2:
          mjlab_ok = mjlab_sps_list[-1] >= 0.5 * mjlab_sps_list[0]
          if not mjlab_ok:
            print("  [FAIL] mjlab scaling: steps/sec did not scale with num_envs.")
          else:
            print("  [OK] mjlab scales with num_envs.")
      print()

  # ---- Verify scaling only (no compare) ----
  if args.verify_scaling and not args.compare_mjlab and results["throughput"]:
    sps = [r["steps_per_sec"] for r in results["throughput"]]
    results["scaling_ok"] = sps[-1] >= 0.5 * sps[0] if len(sps) >= 2 else True
    if not results["scaling_ok"]:
      print("[FAIL] Throughput did not scale with num_envs.")
      return 2
    print("[OK] Throughput scales with num_envs.")

  out = args.output_dir / "latest.json"
  with open(out, "w") as f:
    json.dump(results, f, indent=2)
  print(f"Results written to {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
