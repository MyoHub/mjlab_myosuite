"""
Benchmark: compare raw MyoSuite vs mjlab_myosuite wrapped env (FPS and throughput).

Measures steps per second and rollout throughput for:
  - Raw MyoSuite (single or vectorized via SyncVectorEnv)
  - mjlab_myosuite wrapped env (make_myosuite_env)

Run:
  uv run python examples/benchmark_comparison.py
  uv run python examples/benchmark_comparison.py --plot  # Save comparison plot
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def _has_myosuite() -> bool:
  try:
    import myosuite  # noqa: F401

    return True
  except Exception:
    return False


def benchmark_raw_myosuite(
  env_id: str = "myoElbowPose1D6MRandom-v0",
  num_envs: int = 4,
  num_steps: int = 500,
  warmup_steps: int = 50,
) -> tuple[float, float]:
  """
  Benchmark raw MyoSuite env(s): steps per second and total steps.

  Returns:
    (steps_per_second, total_steps).
  """
  from gymnasium.vector import SyncVectorEnv
  from myosuite.utils import gym as myosuite_gym

  def make_fn():
    return myosuite_gym.make(env_id)

  env = SyncVectorEnv([make_fn for _ in range(num_envs)])
  space = env.single_action_space
  shape = getattr(space, "shape", None)
  action_dim = int(shape[0]) if shape is not None and len(shape) > 0 else 1
  action = np.zeros((num_envs, action_dim), dtype=np.float32)

  # Warmup
  env.reset(seed=0)
  for _ in range(warmup_steps):
    env.step(action)

  # Timed rollout
  env.reset(seed=42)
  t0 = time.perf_counter()
  for _ in range(num_steps):
    env.step(action)
  t1 = time.perf_counter()
  env.close()

  elapsed = t1 - t0
  total_steps = num_steps * num_envs
  return total_steps / elapsed, total_steps


def benchmark_mjlab_myosuite(
  num_envs: int = 4,
  num_steps: int = 500,
  warmup_steps: int = 50,
  device: str = "cpu",
) -> tuple[float, float]:
  """
  Benchmark mjlab_myosuite wrapped env: steps per second and total steps.

  Uses make_myosuite_env so we can set num_envs for a fair comparison with raw MyoSuite.

  Returns:
    (steps_per_second, total_steps).
  """
  from mjlab_myosuite.env_factory import make_myosuite_env

  env = make_myosuite_env("myoElbowPose1D6MRandom-v0", num_envs=num_envs, device=device)
  n = env.num_envs
  single_action = env.action_space.sample()
  if (
    hasattr(single_action, "shape")
    and getattr(single_action, "shape", None) is not None
  ):
    # Batch action: (num_envs, action_dim). sample() may return (n, d) or (d,).
    shape = single_action.shape
    if single_action.ndim >= 2 and shape[0] == n:
      action = np.asarray(single_action, dtype=np.float32)
    else:
      action = np.tile(np.asarray(single_action).reshape(1, -1), (n, 1)).astype(
        np.float32
      )
  else:
    action = (
      np.asarray(single_action, dtype=np.float32)
      if single_action is not None
      else np.zeros((n, 1), dtype=np.float32)
    )

  env.reset(seed=0)
  for _ in range(warmup_steps):
    env.step(action)

  env.reset(seed=42)
  t0 = time.perf_counter()
  for _ in range(num_steps):
    env.step(action)
  t1 = time.perf_counter()
  env.close()

  elapsed = t1 - t0
  total_steps = num_steps * n
  return total_steps / elapsed, total_steps


def run_benchmark(
  num_envs: int = 4,
  num_steps: int = 500,
  warmup_steps: int = 50,
) -> dict:
  """Run both benchmarks and return results dict."""
  raw_sps, raw_total = benchmark_raw_myosuite(
    num_envs=num_envs, num_steps=num_steps, warmup_steps=warmup_steps
  )
  mjlab_sps, mjlab_total = benchmark_mjlab_myosuite(
    num_envs=num_envs,
    num_steps=num_steps,
    warmup_steps=warmup_steps,
  )
  mjlab_gpu_sps, mjlab_gpu_total = benchmark_mjlab_myosuite(
    num_envs=num_envs,
    num_steps=num_steps,
    warmup_steps=warmup_steps,
    device="cuda:0",
  )
  return {
    "raw_myosuite_steps_per_sec": raw_sps,
    "raw_myosuite_total_steps": raw_total,
    "mjlab_myosuite_steps_per_sec": mjlab_sps,
    "mjlab_myosuite_total_steps": mjlab_total,
    "mjlab_myosuite_gpu_steps_per_sec": mjlab_gpu_sps,
    "mjlab_myosuite_gpu_total_steps": mjlab_gpu_total,
    "num_envs": num_envs,
    "num_steps": num_steps,
  }


def save_plot(results: dict, out_path: Path) -> None:
  """Generate a simple comparison bar plot and save to out_path."""
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print(
      "matplotlib not installed; skipping plot. Install with: pip install matplotlib"
    )
    return

  fig, ax = plt.subplots(1, 1, figsize=(6, 4))
  labels = ["Raw MyoSuite", "mjlab_myosuite"]
  sps = [
    results["raw_myosuite_steps_per_sec"],
    results["mjlab_myosuite_steps_per_sec"],
  ]
  colors = ["#2ecc71", "#3498db"]
  bars = ax.bar(labels, sps, color=colors)
  ax.set_ylabel("Steps per second")
  ax.set_title(f"Throughput ({results['num_envs']} envs, {results['num_steps']} steps)")
  for bar, val in zip(bars, sps, strict=True):
    ax.text(
      bar.get_x() + bar.get_width() / 2,
      bar.get_height() + 5,
      f"{val:.0f}",
      ha="center",
    )
  fig.tight_layout()
  fig.savefig(out_path, dpi=150)
  plt.close()
  print(f"Plot saved to {out_path}")


def main():
  parser = argparse.ArgumentParser(description="Benchmark MyoSuite vs mjlab_myosuite")
  parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel envs")
  parser.add_argument("--num-steps", type=int, default=500, help="Steps per benchmark")
  parser.add_argument("--warmup", type=int, default=50, help="Warmup steps")
  parser.add_argument("--plot", action="store_true", help="Save comparison plot")
  parser.add_argument(
    "--out", type=Path, default=Path("benchmark_comparison.png"), help="Plot path"
  )
  args = parser.parse_args()

  if not _has_myosuite():
    print(
      "MyoSuite not installed. Install with: pip install myosuite @ git+https://github.com/MyoHub/myosuite.git@mjx"
    )
    return 1

  print("Running benchmarks (this may take a few seconds)...")
  results = run_benchmark(
    num_envs=args.num_envs,
    num_steps=args.num_steps,
    warmup_steps=args.warmup,
  )

  print("\n--- Results ---")
  print(f"  Raw MyoSuite:      {results['raw_myosuite_steps_per_sec']:.1f} steps/sec")
  print(f"  mjlab_myosuite:    {results['mjlab_myosuite_steps_per_sec']:.1f} steps/sec")
  print(
    f"  mjlab_myosuite GPU: {results['mjlab_myosuite_gpu_steps_per_sec']:.1f} steps/sec"
  )
  print(f"  (num_envs={results['num_envs']}, num_steps={results['num_steps']})")

  if args.plot:
    save_plot(results, args.out)

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
