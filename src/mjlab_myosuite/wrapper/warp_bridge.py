"""Warp bridge: CPU numpy → Warp GPU buffer → PyTorch (mjlab-aligned data path).

When device is CUDA, observations are copied to GPU via Warp and exposed to PyTorch
with zero-copy wp.to_torch(), matching mjlab's "MuJoCo C on CPU, bridged to GPU via
mujoco_warp" architecture.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

_WARP_AVAILABLE: bool | None = None
_wp: Any = None


def _ensure_warp() -> bool:
  """Lazy import of warp; return True if available."""
  global _WARP_AVAILABLE, _wp
  if _WARP_AVAILABLE is not None:
    return _WARP_AVAILABLE
  try:
    import warp as _wp_mod  # noqa: F401

    _wp = _wp_mod
    _WARP_AVAILABLE = True
    return True
  except ImportError:
    _wp = None
    _WARP_AVAILABLE = False
    return False


def is_warp_available() -> bool:
  """Return True if warp is installed and usable."""
  return _ensure_warp()


class WarpObservationBridge:
  """Bridges CPU observation arrays to GPU via Warp and exposes as PyTorch tensors.

  Uses persistent Warp GPU buffers to avoid per-step allocation. One copy
  (CPU → GPU); PyTorch gets a zero-copy view via wp.to_torch().
  """

  def __init__(self, device: str | torch.device):
    if not _ensure_warp():
      raise RuntimeError(
        "Warp is required for WarpObservationBridge; install warp-lang and optionally mujoco-warp."
      )
    self._device_str = str(device).lower()
    if not self._device_str.startswith("cuda"):
      raise ValueError("WarpObservationBridge requires a CUDA device.")
    self._torch_device = torch.device(self._device_str)
    # key -> (wp.array on GPU, last shape)
    self._buffers: dict[str, tuple[Any, tuple[int, ...]]] = {}

  def numpy_to_torch(self, key: str, arr: np.ndarray) -> torch.Tensor:
    """Copy numpy array to GPU via Warp and return a PyTorch tensor (zero-copy view of Warp buffer)."""
    arr = np.asarray(arr, dtype=np.float32)
    shape = arr.shape
    if key not in self._buffers or self._buffers[key][1] != shape:
      gpu_buf = _wp.zeros(shape, dtype=_wp.float32, device=self._device_str)
      self._buffers[key] = (gpu_buf, shape)
    gpu_buf, _ = self._buffers[key]
    cpu_wp = _wp.from_numpy(arr, device="cpu")
    _wp.copy(cpu_wp, gpu_buf)
    return _wp.to_torch(gpu_buf)

  def clear_buffers(self) -> None:
    """Release persistent buffers (e.g. on env close)."""
    self._buffers.clear()
