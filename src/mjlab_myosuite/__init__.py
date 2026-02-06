"""Integration of MyoSuite with mjlab.

- Standard and MJX/Warp MyoSuite environments
- Native mjlab task registration for CLI (train/play)
- Create envs with make_myosuite_env() or make_myosuite_env_from_task_id()
"""

from .env_factory import (
  make_myosuite_env,
  make_myosuite_env_from_task_id,
  task_id_to_myosuite_id,
)
from .registration import get_myosuite_env_ids
from .wrapper import MyoSuiteVecEnvWrapper

try:
  from .tasks import register_myosuite_tasks

  def _register_all():
    try:
      register_myosuite_tasks()
    except Exception:
      pass

except ImportError:

  def _register_all():
    pass


try:
  _register_all()
except Exception as e:
  import warnings

  warnings.warn(
    f"MyoSuite task registration encountered an issue: {e}. "
    "Some tasks may not be available.",
    UserWarning,
    stacklevel=2,
  )

__all__ = [
  "MyoSuiteVecEnvWrapper",
  "get_myosuite_env_ids",
  "make_myosuite_env",
  "make_myosuite_env_from_task_id",
  "task_id_to_myosuite_id",
]
