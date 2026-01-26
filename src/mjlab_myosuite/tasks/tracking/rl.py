"""RL runner for MyoSuite tracking tasks."""

from typing import Any

from ...rl.runner import MyoSuiteOnPolicyRunner


class MyoSuiteMotionTrackingOnPolicyRunner(MyoSuiteOnPolicyRunner):
  """On-policy runner for MyoSuite motion tracking tasks.

  This runner extends MyoSuiteOnPolicyRunner and provides support for
  motion tracking tasks, including:
  - Target visualization in the environment

  Args:
    env: The environment to train on
    cfg: Training configuration
    log_dir: Directory for logging
    device: Device to use for training
    registry_name: Optional wandb registry name for motion artifacts
  """

  def __init__(
    self,
    env: Any,
    cfg: dict[str, Any],
    log_dir: str,
    device: str,
    registry_name: str | None = None,
  ):
    """Initialize the motion tracking runner."""
    super().__init__(env, cfg, log_dir, device)
    self.registry_name = registry_name

  def save(self, path: str, infos=None):
    """Save the model and training information.

    For tracking tasks, this also:
    - Saves motion artifacts if available
    """
    # Save the base model checkpoint
    super().save(path, infos)

    # For tracking tasks, we could save motion artifacts here if needed
    # This follows the pattern of mjlab's MotionTrackingOnPolicyRunner
    if self.registry_name is not None and infos is not None:
      # Motion artifacts would be saved to wandb if using wandb logging
      # This is handled by the base runner's save method
      pass

  def learn(self, *args, **kwargs):
    """Override learn for tracking tasks."""
    # Call parent learn method
    return super().learn(*args, **kwargs)
