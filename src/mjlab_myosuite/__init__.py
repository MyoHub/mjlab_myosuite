"""Auto-registration of MyoSuite environments for mjlab.

This package provides integration between MyoSuite and mjlab, supporting:
- Standard MyoSuite environments
- MJX/Warp compatible versions from the mjx branch
- Native mjlab task registration (when available)
- Gymnasium registry fallback (always works)

"""

# Import registration utilities using relative imports
from .registration import register_myosuite_envs

# Import wrapper using relative imports
from .wrapper import MyoSuiteVecEnvWrapper

# Try to import and use native mjlab task registration
try:
  from .tasks import register_myosuite_tasks

  def _register_all_myosuite_envs():
    """Automatically register all MyoSuite environments."""
    try:
      # Try native mjlab registration first
      register_myosuite_tasks()
    except Exception:
      # Fall back to gymnasium registry on any error
      try:
        register_myosuite_envs()
      except ImportError:
        # MyoSuite not installed, skip registration
        pass

except ImportError:
  # Native task registration not available, use gymnasium registry
  def _register_all_myosuite_envs():
    """Automatically register all MyoSuite environments."""
    try:
      register_myosuite_envs()
    except ImportError:
      # MyoSuite not installed, skip registration
      pass


# Auto-register on import
# Use a try-except to ensure registration completes even if there are warnings
try:
  _register_all_myosuite_envs()
  # Verify registration completed
  import gymnasium as gym

  registered_count = len([k for k in gym.registry.keys() if "Mjlab-MyoSuite" in k])
  if registered_count == 0:
    # Registration didn't complete - try direct registration
    try:
      register_myosuite_envs()
    except Exception:
      pass  # Already tried, skip
except Exception as e:
  # Log but don't fail - registration might have partially completed
  import warnings

  warnings.warn(
    f"MyoSuite environment registration encountered an issue: {e}. "
    "Some environments may not be available.",
    UserWarning,
    stacklevel=2,
  )

__all__ = [
  "MyoSuiteVecEnvWrapper",
  "register_myosuite_envs",
]
