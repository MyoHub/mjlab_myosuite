"""MyoSuite task registration for mjlab."""

from .registration import register_myosuite_tasks

# Note: Auto-registration is handled by the main mjlab_myosuite.__init__ module
# to avoid duplicate registration

__all__ = ["register_myosuite_tasks"]
